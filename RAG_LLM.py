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
from langchain_community.vectorstores import Chroma

# Load Environment Variables (Load MISTRAL API KEY )
load_dotenv()

# --- 1. Page Configuration & Professional UI Layout ---
st.set_page_config(
    page_title="Enterprise RAG Engine",
    page_icon="📄",
    layout="wide"
)

# Custom CSS for Left (User) / Right (Model) chat layout
st.markdown("""
<style>
    .stApp { background-color: #0d1117; }
    
    /* Chat Container */
    .chat-container {
        display: flex;
        flex-direction: column;
        gap: 15px;
        padding-bottom: 25px;
    }
    
    /* User Message (Left Side as requested) */
    .user-msg {
        align-self: flex-start;
        background-color: #2b313e;
        color: white;
        padding: 12px 18px;
        border-radius: 15px 15px 15px 0px;
        max-width: 70%;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        font-size: 15px;
        line-height: 1.5;
    }
    
    /* AI Message (Right Side as requested) */
    .model-msg {
        align-self: flex-end;
        background-color: #1a5c40;
        color: white;
        padding: 12px 18px;
        border-radius: 15px 15px 0px 15px;
        max-width: 75%;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        font-size: 15px;
        line-height: 1.5;
        border: 1px solid #2e7d32;
    }
    
    /* Stats Box in Sidebar */
    .stat-box {
        background-color: #1f242c;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 12px;
        margin-bottom: 10px;
        text-align: center;
        font-size: 14px;
    }
</style>
""", unsafe_allow_html=True)

# --- 2. Caching Strategy to Fix RuntimeError ---

# Embedding model ko alag se ek baar cache karenge taaki httpx client crash na ho
@st.cache_resource
def get_embedding_model():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Main processing pipeline (Passing embedding model with underscore to skip hashing)
@st.cache_resource(show_spinner=False)
def process_pdf_and_create_vectorstore(file_bytes, file_name, _embedding_model):
    # PDF ko read karne ke liye temporary file banana padta hai
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(file_bytes)
        temp_path = temp_file.name

    try:
        # Document Ingestion
        loader = PyPDFLoader(temp_path)
        docs = loader.load()

        # Text Splitting (Chunking)
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100
        )
        chunks = splitter.split_documents(docs)

        # Vector Database Indexing (FIXED: passing chunks here, not docs)
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=_embedding_model,
            persist_directory=f'Chroma_DB_{file_name.replace(".pdf", "")}'
        )
        
        # MMR Retriever Setup
        retriever = vectorstore.as_retriever(
            search_type='mmr',
            search_kwargs={
                "k": 4,
                "fetch_k": 10,
                "lambda_mult": 0.5
            }
        )
        return retriever, len(chunks), len(docs)
    
    finally:
        # Cleanup temp file after usage
        if os.path.exists(temp_path):
            os.remove(temp_path)

# --- 3. Sidebar Panel (Uploads & Logs) ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/8132/8132544.png", width=60)
    st.title("📄 RAG Control Center")
    st.markdown("### Project: Document Intelligence System")
    st.caption("Tech Stack: LangChain | Chroma DB | HuggingFace | Mistral")
    st.divider()
    
    # Dynamic PDF file upload control
    uploaded_file = st.file_uploader("Upload Target PDF Document", type=["pdf"])
    
    st.divider()
    st.markdown("### 📊 Engine Metrics")
    
    if not uploaded_file:
        st.info("Awaiting Document Ingestion...")

# --- 4. Main App Logic ---
st.title("⚡ AI-Powered Semantic Search Engine")
st.caption("Perform accurate, hallucination-free Q&A over complex or enterprise documents.")

# Initialize Session State for Chat Records
if "rag_history" not in st.session_state:
    st.session_state.rag_history = []

if uploaded_file:
    # Safely load the cached embedding model
    embedding_model = get_embedding_model()
    
    # Trigger RAG core vector indexing pipeline
    with st.spinner("🧠 Initializing Semantic Indexing & Vector Embeddings..."):
        retriever, total_chunks, total_pages = process_pdf_and_create_vectorstore(
            uploaded_file.getvalue(), uploaded_file.name, _embedding_model=embedding_model
        )
    
    # Populate structural metrics in the sidebar
    with st.sidebar:
        st.success("Document Ingested Successfully!")
        st.markdown(f'<div class="stat-box">🎒 <b>Total Pages:</b> {total_pages}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="stat-box">🧩 <b>Vector Chunks:</b> {total_chunks}</div>', unsafe_allow_html=True)
        
        if st.button("🗑️ Clear Context & Chat"):
            st.session_state.rag_history = []
            st.rerun()

    # Render Active Chat Board
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    for msg in st.session_state.rag_history:
        if msg["role"] == "user":
            st.markdown(f'<div class="user-msg"><b>You:</b><br>{msg["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="model-msg"><b>System Response:</b><br>{msg["content"]}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # User query input line
    if user_query := st.chat_input("Ask a question about the document..."):
        # Instantly log user prompt to history and refresh UI
        st.session_state.rag_history.append({"role": "user", "content": user_query})
        st.rerun()

    # Execute backend search and prediction if last message is from user
    if st.session_state.rag_history and st.session_state.rag_history[-1]["role"] == "user":
        active_query = st.session_state.rag_history[-1]["content"]
        
        with st.spinner("🔍 Executing Semantic Search & Context Synthesis..."):
            # 1. Retrieve raw text contexts matching the query semantic space
            retrieved_chunks = retriever.invoke(active_query)
            context_string = "\n\n".join([doc.page_content for doc in retrieved_chunks])
            
            # 2. Perfected Professional Anti-Hallucination Prompt Template
            prompt_template = ChatPromptTemplate.from_messages([
                ("system", """You are an advanced document intelligence assistant. Your sole task is to answer the user's question accurately using only the retrieved source context provided below.
                
                Strict Operation Protocols:
                1. Rely strictly on the explicit facts presented in the Context. Do not extrapolate, assume, or use external training knowledge.
                2. If the context does not provide enough information to answer the question, respond exactly with: "I am sorry, but the answer cannot be found in the uploaded document." 
                3. Keep the output professionally structured, direct, and factually airtight.
                
                Retrieved Context:\n{context}"""),
                ("human", "{question}")
            ])
            
            # 3. Format inputs and invoke deterministic LLM pipeline
            formatted_messages = prompt_template.format_messages(context=context_string, question=active_query)
            model = ChatMistralAI(model="mistral-small-2603", temperature=0.0) # 0.0 prevents random variations
            ai_response = model.invoke(formatted_messages)
            
            # 4. Save to history and update application state
            st.session_state.rag_history.append({"role": "assistant", "content": ai_response.content})
            st.rerun()
else:
    # Default initial landing frame
    st.info("👈 Please upload a PDF document in the sidebar to initialize the vector database and begin testing.")
    
    st.markdown("""
    ---
    ### 🛠️ Core Production Engineering Details:
    *   **Deterministic Synthesis:** The engine operates at `temperature=0.0` bound by strict system-level instructions to safeguard against artificial hallucination.
    *   **Maximal Marginal Relevance (MMR):** The chunk selection query layer prevents context window flooding by choosing documents that balance raw similarity score with topical diversity.
    *   **Zero-Cost Native Embeddings:** Computes textual vector coordinates locally using the optimized open-source `all-MiniLM-L6-v2` transformer architecture.
    """)
