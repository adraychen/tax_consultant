import streamlit as st
import os
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.vector_stores.supabase import SupabaseVectorStore
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

load_dotenv()

st.title("🇨🇦 Canada Tax Assistant")

gemini_key = st.secrets.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY"))
supabase_db_url = st.secrets.get("SUPABASE_DB_URL", os.getenv("SUPABASE_DB_URL"))

if not gemini_key or not supabase_db_url:
    st.error("Missing configuration keys.")
    st.stop()

try:
    llm = GoogleGenAI(model="gemini-2.5-flash", api_key=gemini_key)
except Exception as e:
    st.error(f"Model initialization failed: {str(e)}")
    st.stop()

embed_model = HuggingFaceEmbedding(model_name="nlpaueb/legal-bert-base-uncased")

vector_store = SupabaseVectorStore(
    postgres_connection_string=supabase_db_url,
    collection_name="tax_knowledge"
)
index = VectorStoreIndex.from_vector_store(vector_store, embed_model=embed_model)

vector_retriever = index.as_retriever(similarity_top_k=5)
bm25_retriever = BM25Retriever.from_defaults(index=index, similarity_top_k=5)

retriever = QueryFusionRetriever(
    [vector_retriever, bm25_retriever],
    similarity_top_k=5,
    num_queries=1,
    mode="reciprocal_rerank"
)

query_engine = RetrieverQueryEngine.from_args(retriever=retriever, llm=llm)

if prompt := st.chat_input("How can I help with your tax question?"):
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        try:
            response = query_engine.query(prompt)
            st.markdown(response.response)
        except Exception as e:
            st.error(f"Query failed: {str(e)}")
