import os
import re
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.readers.web import BeautifulSoupWebReader
from llama_index.vector_stores.supabase import SupabaseVectorStore
from google import genai
from typing import List
import time
# load_dotenv()

# Custom Gemini Embedding class using google-genai SDK directly
class GeminiEmbedding(BaseEmbedding):
    _client: any = PrivateAttr()
    _model: str = PrivateAttr()

    def __init__(self, api_key: str, model: str = "models/gemini-embedding-2", **kwargs):
        super().__init__(**kwargs)
        self._client = genai.Client(api_key=api_key)
        self._model = model

    def _get_text_embedding(self, text: str) -> List[float]:
        response = self._client.models.embed_content(model=self._model, contents=text)
        return response.embeddings[0].values

    # def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
    #     return [self._get_text_embedding(text) for text in texts]

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        results = []
        for text in texts:
            results.append(self._get_text_embedding(text))
            time.sleep(0.5)  # 0.5 second delay between requests
        return results

    def _get_query_embedding(self, query: str) -> List[float]:
        return self._get_text_embedding(query)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        return self._get_text_embedding(text)

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return self._get_query_embedding(query)

    async def _aget_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        return self._get_text_embeddings(texts)


def extract_tax_act_metadata(text):
    """Extract section metadata from Income Tax Act text."""
    metadata = {}

    part_match = re.search(r'PART\s+([IVX]+(?:\.[A-Z0-9]+)?)', text)
    if part_match:
        metadata["part"] = f"PART {part_match.group(1)}"

    div_match = re.search(r'DIVISION\s+([A-Z](?:\.[A-Z0-9]+)?)', text)
    if div_match:
        metadata["division"] = f"DIVISION {div_match.group(1)}"

    marginal_match = re.search(r'Marginal note:([^\n]+)', text)
    if marginal_match:
        metadata["section_title"] = marginal_match.group(1).strip()

    section_match = re.search(r'\b(\d{1,3}(?:\.\d+)?)\s*[\(\s]', text)
    if section_match:
        metadata["section_number"] = section_match.group(1)

    return metadata


def enrich_nodes(nodes, source_name, is_tax_act=True):
    """Add source and section metadata to nodes."""
    for node in nodes:
        node.metadata["source"] = source_name
        if is_tax_act:
            node.metadata.update(extract_tax_act_metadata(node.text))
    return nodes


gemini_key = "AIzaSyA9T-Li31yOi8HzFLwx0m4MD8JLax1QfWM"
# gemini_key = os.getenv("GEMINI_API_KEY")
supabase_db_url = "postgresql://postgres.dmphfihfkvrxegqcbufc:uzId5VyZQNH2LgLA@aws-1-us-east-2.pooler.supabase.com:5432/postgres"
# supabase_db_url = os.getenv("SUPABASE_DB_URL")

embed_model = GeminiEmbedding(api_key=gemini_key)

vector_store = SupabaseVectorStore(
    postgres_connection_string=supabase_db_url,
    collection_name="tax_knowledge"
)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

loader = BeautifulSoupWebReader()
splitter = SentenceSplitter(chunk_size=512, chunk_overlap=50)

# Load Income Tax Act
print("Loading Income Tax Act...")
tax_act_docs = loader.load_data(urls=["https://laws-lois.justice.gc.ca/eng/acts/i-3.3/FullText.html"])
tax_act_nodes = splitter.get_nodes_from_documents(tax_act_docs)
tax_act_nodes = enrich_nodes(tax_act_nodes, "Income Tax Act", is_tax_act=True)
print(f"Created {len(tax_act_nodes)} chunks from Income Tax Act.")

# Load CRA plain language guides
print("Loading CRA guides...")
cra_urls = [
    "https://www.canada.ca/en/revenue-agency/services/tax/businesses/topics/sole-proprietorships-partnerships/business-expenses.html",
    "https://www.canada.ca/en/revenue-agency/services/tax/businesses/topics/sole-proprietorships-partnerships/business-expenses/motor-vehicle-expenses.html",
    "https://www.canada.ca/en/revenue-agency/services/tax/businesses/topics/sole-proprietorships-partnerships/business-expenses/home-office-expenses.html",
    "https://www.canada.ca/en/revenue-agency/services/tax/businesses/topics/sole-proprietorships-partnerships/business-expenses/hire-family-members.html",
]
cra_docs = loader.load_data(urls=cra_urls)
cra_nodes = splitter.get_nodes_from_documents(cra_docs)
cra_nodes = enrich_nodes(cra_nodes, "CRA Guide", is_tax_act=False)
print(f"Created {len(cra_nodes)} chunks from CRA guides.")


# Combine and ingest in batches
all_nodes = tax_act_nodes + cra_nodes
print(f"Total chunks to ingest: {len(all_nodes)}")

BATCH_SIZE = 50

# From start to end

# for i in range(0, len(all_nodes), BATCH_SIZE):
#     batch = all_nodes[i:i + BATCH_SIZE]
#
#     # Fresh connection for each batch
#     vector_store = SupabaseVectorStore(
#         postgres_connection_string=supabase_db_url,
#         collection_name="tax_knowledge"
#     )
#     storage_context = StorageContext.from_defaults(vector_store=vector_store)
#
#     VectorStoreIndex(batch, storage_context=storage_context, embed_model=embed_model)
#     print(
#         f"Ingested batch {i // BATCH_SIZE + 1} of {len(all_nodes) // BATCH_SIZE + 1} ({i + len(batch)}/{len(all_nodes)} chunks)")

# Resume from chunk 2200
START_FROM = 2200

remaining_nodes = all_nodes[START_FROM:]
print(f"Resuming from chunk {START_FROM}. Remaining: {len(remaining_nodes)} chunks.")

for i in range(0, len(remaining_nodes), BATCH_SIZE):
    batch = remaining_nodes[i:i + BATCH_SIZE]

    vector_store = SupabaseVectorStore(
        postgres_connection_string=supabase_db_url,
        collection_name="tax_knowledge"
    )
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    VectorStoreIndex(batch, storage_context=storage_context, embed_model=embed_model)

    completed = START_FROM + i + len(batch)
    print(f"Ingested batch {i // BATCH_SIZE + 1} ({completed}/{len(all_nodes)} chunks)")

    time.sleep(2)  # Increased to 2 seconds between batches to avoid rate limit

print("Ingestion complete.")
