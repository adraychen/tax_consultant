import streamlit as st
import os
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex
from llama_index.core.query_engine import SubQuestionQueryEngine
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.vector_stores.supabase import SupabaseVectorStore
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.question_gen import LLMQuestionGenerator

load_dotenv()

st.set_page_config(page_title="Tax Assistant", page_icon="🇨🇦")
st.title("🇨🇦 Canada Tax Assistant")

gemini_key = st.secrets.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY"))
supabase_db_url = st.secrets.get("SUPABASE_DB_URL", os.getenv("SUPABASE_DB_URL"))

if not gemini_key or not supabase_db_url:
    st.error("Missing configuration keys. Please check Streamlit Secrets or your .env file.")
    st.stop()

@st.cache_resource
def init_models():
    llm = GoogleGenAI(model="gemini-2.5-flash", api_key=gemini_key)
    embed_model = HuggingFaceEmbedding(model_name="nlpaueb/legal-bert-base-uncased")
    return llm, embed_model

# @st.cache_resource
# def init_query_engine(_llm, _embed_model):
#     vector_store = SupabaseVectorStore(
#         postgres_connection_string=supabase_db_url,
#         collection_name="tax_knowledge"
#     )
#     index = VectorStoreIndex.from_vector_store(vector_store, embed_model=_embed_model)

#     base_engine = index.as_query_engine(llm=_llm, similarity_top_k=15)

#     tools = [
#         QueryEngineTool(
#             query_engine=base_engine,
#             metadata=ToolMetadata(
#                 name="canada_income_tax_act",
#                 description="Contains the full text of the Canadian Income Tax Act including rules on income, deductions, credits, penalties, corporate tax, non-residents, and anti-avoidance provisions."
#             )
#         )
#     ]

#     return SubQuestionQueryEngine.from_defaults(
#         query_engine_tools=tools,
#         llm=_llm,
#         verbose=False
#     )


@st.cache_resource
def init_query_engine(_llm, _embed_model):
    vector_store = SupabaseVectorStore(
        postgres_connection_string=supabase_db_url,
        collection_name="tax_knowledge"
    )
    index = VectorStoreIndex.from_vector_store(vector_store, embed_model=_embed_model)

    base_engine = index.as_query_engine(llm=_llm, similarity_top_k=15)

    tools = [
        QueryEngineTool(
            query_engine=base_engine,
            metadata=ToolMetadata(
                name="canada_income_tax_act",
                description="Contains the full text of the Canadian Income Tax Act including rules on income, deductions, credits, penalties, corporate tax, non-residents, and anti-avoidance provisions."
            )
        )
    ]

    question_gen = LLMQuestionGenerator.from_defaults(llm=_llm)

    return SubQuestionQueryEngine.from_defaults(
        query_engine_tools=tools,
        llm=_llm,
        question_gen=question_gen,
        verbose=False
    )
    
llm, embed_model = init_models()
query_engine = init_query_engine(llm, embed_model)

st.caption("⚠️ This app is a prototype and does not provide professional tax advice. Always consult a qualified tax professional.")

if prompt := st.chat_input("How can I help with your tax question?"):
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Searching tax regulations..."):
            try:
                response = query_engine.query(prompt)
                st.markdown(response.response)
            except Exception as e:
                st.error(f"Query failed: {str(e)}")
