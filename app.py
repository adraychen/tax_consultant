import streamlit as st
import os
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.supabase import SupabaseVectorStore
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding

load_dotenv()

st.title("🇨🇦 Canada Tax Assistant")

gemini_key = st.secrets.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY"))
supabase_db_url = st.secrets.get("SUPABASE_DB_URL", os.getenv("SUPABASE_DB_URL"))

if not gemini_key or not supabase_db_url:
    st.error("Missing configuration keys.")
    st.stop()

try:
    llm = GoogleGenAI(model="gemini-2.0-flash", api_key=gemini_key)
except Exception as e:
    st.error(f"LLM init failed: {str(e)}")
    st.stop()

embed_model = GoogleGenAIEmbedding(model_name="text-embedding-004", api_key=gemini_key)

try:
    vector_store = SupabaseVectorStore(
        postgres_connection_string=supabase_db_url,
        collection_name="tax_knowledge"
    )
except Exception as e:
    st.error(f"Supabase connection failed: {str(e)}")
    st.stop()

index = VectorStoreIndex.from_vector_store(vector_store, embed_model=embed_model)
query_engine = index.as_query_engine(llm=llm)

if prompt := st.chat_input("How can I help with your tax question?"):
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        response = query_engine.query(prompt)
        st.markdown(response.response)
