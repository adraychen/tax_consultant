import os
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext
from llama_index.vector_stores.supabase import SupabaseVectorStore
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding

load_dotenv()

def run_ingestion():
    gemini_key = os.getenv("GEMINI_API_KEY")
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_KEY")

    embed_model = GoogleGenAIEmbedding(model_name="text-embedding-004", api_key=gemini_key)

    vector_store = SupabaseVectorStore(
        postgres_connection_string=f"postgresql://postgres:{supabase_key}@{supabase_url.split('//')[1]}/postgres",
        collection_name="tax_knowledge"
    )
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    documents = SimpleDirectoryReader("./data").load_data()
    index = VectorStoreIndex.from_documents(documents, storage_context=storage_context, embed_model=embed_model)
    print("Ingestion complete.")

if __name__ == "__main__":
    run_ingestion()
