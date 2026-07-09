import streamlit as st
import os
import tempfile
from dotenv import load_dotenv

# LangChain & Mistral Imports
from langchain_mistralai import ChatMistralAI 
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
# Tavily Search Tool Import
from langchain_community.tools.tavily_search import TavilySearchResults

# Load Environment Variables
load_dotenv()

# --- 1. Page Configuration ---
st.set_page_config(
    page_title="Enterprise RAG Engine",
    page_icon="📄",
    layout="wide"
)

# --- 2. Dynamic Theme Toggle (Ab default "Light" rahega) ---
if "theme" not in st.session_state:
    st.session_state.theme = "Light"  # Default set to Light Mode

with st.sidebar:
    st.markdown("### 🎨 UI Customization")
    # Toggle switch active rahega agar theme Light hai (jo ki default hai)
    theme_toggle = st.toggle("🌙 Light Mode Active", value=(st.session_state.theme == "Light"))
    st.session_state.theme = "Light" if theme_toggle else "Dark"
    st.divider()

if st.session_state.theme == "Dark":
    bg_color = "#0d1117"
    text_color = "#ffffff"
    user_bg = "#2b313e"
    model_bg = "#1a5c40"
    model_border = "#2e7d32"
    stat_bg = "#1f242c"
    border_color = "#30363d"
else:
    bg_color = "#f4f6f9"
    text_color = "#111111"
    user_bg = "#e3f2fd"
    model_bg = "#e8f5e9"
    model_border = "#a5d6a7"
    stat_bg = "#ffffff"
    border_color = "#d1d5db"

st.markdown(f"""
<style>
    .stApp {{ background-color: {bg_color}; color: {text_color}; }}
    .chat-container {{ display: flex; flex-direction: column; gap: 15px; padding-bottom: 25px; }}
    .user-msg {{
        align-self: flex-start; background-color: {user_bg}; color: {text_color} !important;
        padding: 12px 18px; border-radius: 15px 15px 15px 0px; max-width: 70%;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05); font-size: 15px; line-height: 1.5;
    }}
    .model-msg {{
        align-self: flex-end; background-color: {model_bg}; color: {text_color} !important;
        padding: 12px 18px; border-radius: 15px 15px 0px 15px; max-width: 75%;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05); font-size: 15px; line-height: 1.5; border: 1px solid {model_border};
    }}
    .stat-box {{
        background-color: {stat_bg}; border: 1px solid {border_color}; color: {text_color};
        border-radius: 8px; padding: 12px; margin-bottom: 10px; text-align: center; font-size: 14px;
    }}
</style>
""", unsafe_allow_html=True)

# --- 3. Caching Embedding Model & Initializing Tavily Search ---
@st.cache_resource
def get_embedding_model():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

@st.cache_resource
def get_search_tool():
    # Tavily search initialization with 3 max results
    return TavilySearchResults(max_results=3)

def process_pdf_with_validation(file_bytes, _embedding_model):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(file_bytes)
        temp_path = temp_file.name

    try:
        loader = PyPDFLoader(temp_path)
        docs = loader.load()

        extracted_text_sample = "".join([doc.page_content for doc in docs]).strip()
        if not extracted_text_sample:
            return "EMPTY_TEXT", 0, 0

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(docs)

        vectorstore = FAISS.from_documents(documents=chunks, embedding=_embedding_model)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
        
        return retriever, len(chunks), len(docs)
    
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

# --- 4. Sidebar Control Panel ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/8132/8132544.png", width=60)
    st.title("📄 RAG Control Center")
    st.caption("Tech Stack: LangChain | FAISS | Mistral AI | Tavily")
    st.divider()
    
    uploaded_file = st.file_uploader("Upload Target PDF Document", type=["pdf"])
    st.divider()
    st.markdown("### 📊 Engine Metrics")

# --- 5. Main App Logic ---
st.title("⚡ AI-Powered Semantic Search Engine")

if "rag_history" not in st.session_state:
    st.session_state.rag_history = []
if "cached_retriever" not in st.session_state:
    st.session_state.cached_retriever = None
if "current_file" not in st.session_state:
    st.session_state.current_file = None
if "metrics" not in st.session_state:
    st.session_state.metrics = {"chunks": 0, "pages": 0}

if uploaded_file:
    if st.session_state.current_file != uploaded_file.name:
        embedding_model = get_embedding_model()
        with st.spinner("🧠 Extracting Text and Building Semantic Map..."):
            retriever, total_chunks, total_pages = process_pdf_with_validation(
                uploaded_file.getvalue(), embedding_model
            )
            
            if retriever == "EMPTY_TEXT":
                st.session_state.cached_retriever = "ERROR"
                st.session_state.metrics = {"chunks": 0, "pages": 0}
            else:
                st.session_state.cached_retriever = retriever
                st.session_state.metrics["chunks"] = total_chunks
                st.session_state.metrics["pages"] = total_pages
                st.session_state.rag_history = []
            st.session_state.current_file = uploaded_file.name

    with st.sidebar:
        if st.session_state.cached_retriever == "ERROR":
            st.error("❌ Text Extraction Failed!")
        else:
            st.success("Document Ingested Successfully!")
            st.markdown(f'<div class="stat-box">🎒 <b>Total Pages:</b> {st.session_state.metrics["pages"]}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="stat-box">🧩 <b>Vector Chunks:</b> {st.session_state.metrics["chunks"]}</div>', unsafe_allow_html=True)
        
        if st.button("🗑️ Clear Context & Chat"):
            st.session_state.rag_history = []
            st.session_state.cached_retriever = None
            st.session_state.current_file = None
            st.session_state.metrics = {"chunks": 0, "pages": 0}
            st.rerun()

    if st.session_state.cached_retriever == "ERROR":
        st.error("⚠️ Is PDF se text nahi nikal pa raha hai.")
    else:
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        for msg in st.session_state.rag_history:
            if msg["role"] == "user":
                st.markdown(f'<div class="user-msg"><b>You:</b><br>{msg["content"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="model-msg"><b>System Response:</b><br>{msg["content"]}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        if user_query := st.chat_input("Ask a question about the document..."):
            st.session_state.rag_history.append({"role": "user", "content": user_query})
            st.rerun()

        if st.session_state.rag_history and st.session_state.rag_history[-1]["role"] == "user":
            active_query = st.session_state.rag_history[-1]["content"]
            
            with st.spinner("🔍 Analyzing document context..."):
                if st.session_state.cached_retriever not in [None, "ERROR"]:
                    # Document Context retrieve karein
                    retrieved_chunks = st.session_state.cached_retriever.invoke(active_query)
                    context_string = "\n\n".join([doc.page_content for doc in retrieved_chunks])
                    
                    # --- Step 1: Prompt updated for hybrid operations ---
                    prompt_template = ChatPromptTemplate.from_messages([
                        ("system", """You are an advanced document intelligence assistant. You are currently analyzing an uploaded document named '{filename}'.
                        
                        Strict Operation Protocols:
                        1. GREETINGS & META-QUESTIONS: If the user greets you (e.g., 'hi', 'hello', 'hey') or asks about the document status (e.g., 'kya pdf mila', 'is the file uploaded'), respond naturally, politely, and confirm that the file '{filename}' is ready.
                        2. CONTEXT ANALYSIS: For any content question, check the Context below. If found, answer strictly using the context facts without extrapolation.
                        3. MISSING INFO PROTOCOL: If the user asks a specific factual question and the context completely lacks that information, respond exactly with this token: "NOT_FOUND_IN_DOC". Do not try to answer using your pre-trained knowledge if it looks like a specific query meant for the document.
                        
                        Context:\n{context}"""),
                        ("human", "{question}")
                    ])
                    
                    formatted_messages = prompt_template.format_messages(
                        context=context_string, 
                        question=active_query,
                        filename=st.session_state.current_file
                    )
                    
                    model = ChatMistralAI(model="mistral-small-latest", temperature=0.0) 
                    ai_response = model.invoke(formatted_messages)
                    response_content = ai_response.content
                    
                    # --- Step 2: Fallback to Tavily Web Search if not found in PDF ---
                    if "NOT_FOUND_IN_DOC" in response_content:
                        with st.spinner("🌐 Looking up the web for accurate info..."):
                            search_tool = get_search_tool()
                            search_results = search_tool.invoke({"query": active_query})
                            
                            # Tavily context ke sath model ko firse run karein
                            fallback_template = ChatPromptTemplate.from_messages([
                                ("system", """You are a helpful AI Assistant. The user's question was not present in their uploaded document, so we searched the live web.
                                Based on the Web Search Results provided below, give a professional and detailed answer to the user's question. Inform them lightly that this info is from the web.
                                
                                Web Search Results:\n{web_context}"""),
                                ("human", "{question}")
                            ])
                            
                            fallback_messages = fallback_template.format_messages(
                                web_context=str(search_results),
                                question=active_query
                            )
                            ai_fallback_response = model.invoke(fallback_messages)
                            response_content = ai_fallback_response.content
                else:
                    response_content = "⚠️ Core database context missing. Please re-upload the document."
                
                st.session_state.rag_history.append({"role": "assistant", "content": response_content})
                st.rerun()
else:
    with st.sidebar:
        st.info("Awaiting Document Ingestion...")
    st.info("👈 Please upload a PDF document in the sidebar to initialize the engine.")
