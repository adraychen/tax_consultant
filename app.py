import streamlit as st
import os
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.supabase import SupabaseVectorStore
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# Force load environment variables for local testing
load_dotenv()

st.set_page_config(page_title="Canada Tax Assistant", page_icon="🇨🇦")
st.title("🇨🇦 Canada Tax Assistant")

# Prioritize Streamlit Secrets, fallback to environment variables
gemini_key = st.secrets.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY"))
supabase_db_url = st.secrets.get("SUPABASE_DB_URL", os.getenv("SUPABASE_DB_URL"))

if not gemini_key or not supabase_db_url:
    st.error("Missing configuration keys. Please check Streamlit Secrets or your .env file.")
    st.stop()

@st.cache_resource
def init_models():
    try:
        # Switching to the explicit 'latest' string often resolves 404/NOT_FOUND errors 
        # in the llama-index google-genai integration.
        llm = GoogleGenAI(model="models/gemini-2.0-flash", api_key=gemini_key)
        
        # mpnet-base-v2 is a robust choice for tax-related semantic search.
        embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-mpnet-base-v2")
        
        return llm, embed_model
    except Exception as e:
        st.error(f"Model initialization failed: {str(e)}")
        return None, None

llm, embed_model = init_models()

if llm and embed_model:
    try:
        # Initialize Supabase Vector Store
        vector_store = SupabaseVectorStore(
            postgres_connection_string=supabase_db_url,
            collection_name="tax_knowledge"
        )
        
        # Build the index
        index = VectorStoreIndex.from_vector_store(vector_store, embed_model=embed_model)
        query_engine = index.as_query_engine(llm=llm)
        
    except Exception as e:
        st.error(f"Vector Store connection failed: {str(e)}")
        st.stop()

    if prompt := st.chat_input("How can I help with your tax question?"):
        with st.chat_message("user"):
            st.markdown(prompt)
            
        with st.chat_message("assistant"):
            try:
                with st.spinner("Searching tax regulations..."):
                    response = query_engine.query(prompt)
                    st.markdown(response.response)
            except Exception as e:
                # If it still fails with a 429, it means the specific API key 
                # has no quota assigned in the Google Cloud Console.
                st.error(f"Query failed: {str(e)}")
