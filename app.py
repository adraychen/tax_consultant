import streamlit as st
import os
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.supabase import SupabaseVectorStore
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.google import GooglePaLMEmbedding

load_dotenv()

st.title("🇨🇦 Canada Tax Assistant")

gemini_key = st.secrets.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY"))
supabase_url = st.secrets.get("SUPABASE_URL", os.getenv("SUPABASE_URL"))
supabase_key = st.secrets.get("SUPABASE_KEY", os.getenv("SUPABASE_KEY"))

if not gemini_key or not supabase_url:
    st.error("Missing configuration keys.")
    st.stop()

llm = GoogleGenAI(model="models/gemini-1.5-flash", api_key=gemini_key)
embed_model = GooglePaLMEmbedding(model_name="models/embedding-001", api_key=gemini_key)

vector_store = SupabaseVectorStore(
    postgres_connection_string=f"postgresql://postgres:{supabase_key}@{supabase_url.split('//')[1]}/postgres",
    collection_name="tax_knowledge"
)
index = VectorStoreIndex.from_vector_store(vector_store, embed_model=embed_model)
query_engine = index.as_query_engine(llm=llm)

if prompt := st.chat_input("How can I help with your tax question?"):
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        response = query_engine.query(prompt)
        st.markdown(response.response)
