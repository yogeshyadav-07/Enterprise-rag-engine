# Enterprise Document Intelligence System (RAG Engine)

A high-performance, deterministic **Retrieval-Augmented Generation (RAG)** pipeline that enables accurate, hallucination-free Q&A over complex or dense corporate and academic PDF documents. Built using Python, Streamlit, LangChain, Chroma DB, and Mistral AI.

---

## 🚀 Key Features

*   **Dynamic PDF Ingestion:** Upload any PDF document via the intuitive sidebar to automatically parse and index its contents on the fly.
*   **Optimal Chunking Strategy:** Implements `RecursiveCharacterTextSplitter` with balanced chunk sizes and overlaps to retain deep semantic context across boundaries.
*   **Local High-Performance Embeddings:** Leverages the open-source `all-MiniLM-L6-v2` transformer model locally via Hugging Face, completely neutralizing runtime vector computation costs.
*   **Maximal Marginal Relevance (MMR):** Advanced context retrieval layer that optimizes for both query relevance and topical diversity, preventing context-window flooding.
*   **Deterministic Guardrails:** Hard-coded system prompt boundaries paired with a `temperature=0.0` configuration completely eliminate AI hallucinations, ensuring answers are 100% grounded in source facts.
*   **Custom Interface:** Fully tailored asynchronous conversational UI featuring clean state management and isolated user/system messaging spaces.

---

## 🛠️ Tech Stack Architecture

| Component | Technology | Role |
|---|---|---|
| **Frontend Framework** | Streamlit | UI Workspace & Session Management |
| **Orchestration Layer** | LangChain | Prompt Engineering & Retrieval Pipelines |
| **Vector Storage** | Chroma DB | Embedded Vector Index & Metadata Database |
| **Text Embedding** | Hugging Face (`all-MiniLM-L6-v2`) | Local High-Dimensional Coordinate Generation |
| **Large Language Model** | Mistral AI (`mistral-small-2603`) | Context-Bound Conversational Synthesis |

---

## 📋 System Setup & Installation

Follow these quick steps to get the environment configured and the dashboard running locally.

### 1. Clone the Repository
```bash
git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
cd your-repo-name

```

### 2. Install Dependencies

Make sure you have Python installed, then execute:

```bash
pip install -r requirements.txt

```

### 3. Environment Configuration

Create a `.env` file in the root directory of your project (referencing the provided `.env.example`) and append your secure API token:

```text
MISTRAL_API_KEY=your_actual_mistral_api_key_here

```

### 4. Boot the Application Server

Run the Streamlit server locally:

```bash
streamlit run RAG.py

```

---

## 🧠 Architecture Flow Detail

1. **Document Loading:** The target document is extracted using `PyPDFLoader`.
2. **Semantic Text Splitting:** Documents are fragmented into chunks of 1000 characters with a 100-character sliding overlap.
3. **Vector Mapping & Storage:** Text fragments are vectorized and locally cached inside a persistent `Chroma` instance.
4. **Context-Aware Query Routing:** The MMR algorithm fetches the top 4 highly distinct relevant fragments.
5. **Prompt Synthesis & LLM Inference:** System context constraints are programmatically bound with user input and processed deterministically by Mistral Small.

```

### Is README me kya khaas hai?
1. **Industry Keywords:** Isme **Deterministic Guardrails**, **Semantic Boundaries**, **Context-Aware Routing**, aur **Context-Window Flooding** jaise software development terms use kiye hain, jo dikhate hain ki aapko sirf code chalana nahi aata, balki architectural concepts bhi pata hain.
2. **Clear Layout:** Table aur clean lists ka use kiya hai taaki recruiters ko ek nazar mein samajh aa jaye ki aapne kaun-kaunsi technologies use ki hain.

Aapki teeno files (`app.py`, `requirements.txt`, `.env.example`) aur yeh `README.md` jab ek sath GitHub par jayengi, toh aapka profile ekdum professional lagega!

```
