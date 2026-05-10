# 🦜 LangChain RAG with Multiple Vector Databases

A comprehensive implementation of **Retrieval Augmented Generation (RAG)** using LangChain, Groq LLM, and multiple vector databases.

---

## 📌 What is RAG?

**RAG (Retrieval Augmented Generation)** is a technique that combines:
- **Retrieval** → fetches relevant documents from a vector database
- **Augmented** → adds retrieved documents as context to the prompt
- **Generation** → LLM generates answer based on the context

```
User Question
      ↓
Retriever → searches Vector DB → finds relevant chunks
      ↓
Prompt → fills {context} + {input}
      ↓
LLM → generates answer based on context
      ↓
Final Answer ✅
```

---

## ⚙️ Tech Stack

| Component | Technology |
|---|---|
| **Framework** | LangChain v1.x |
| **LLM** | Groq (llama-3.3-70b-versatile) |
| **Embeddings** | Ollama (llama3.2) |
| **Vector DBs** | FAISS, Chroma, ObjectBox |
| **UI** | Streamlit |
| **Document Loader** | PyPDFDirectoryLoader, WebBaseLoader |

---

## 🗃️ Vector Databases Compared

### 1. FAISS (Facebook AI Similarity Search)
```python
from langchain_community.vectorstores import FAISS

vectors = FAISS.from_documents(docs, embeddings)
retriever = vectors.as_retriever(search_kwargs={"k": 3})
```

### 2. Chroma
```python
from langchain_community.vectorstores import Chroma

vectors = Chroma.from_documents(docs, embeddings)
retriever = vectors.as_retriever(search_kwargs={"k": 3})
```

### 3. ObjectBox
```python
from langchain_community.vectorstores import ObjectBox

vectors = ObjectBox.from_documents(
    docs,
    embeddings,
    embedding_dimensions=4096  # required!
)
retriever = vectors.as_retriever(search_kwargs={"k": 3})
```


## 🔄 RAG Pipeline

```
Documents (PDF/Web)
        ↓
Document Loader
        ↓
RecursiveCharacterTextSplitter
(chunk_size=1000, chunk_overlap=200)
        ↓
OllamaEmbeddings (llama3.2)
        ↓
Vector Database (FAISS/Chroma/ObjectBox)
        ↓
Retriever
        ↓
Prompt Template
        ↓
Groq LLM (llama-3.3-70b-versatile)
        ↓
StrOutputParser
        ↓
Final Answer ✅
```

---

## 📦 Installation

```bash
# Clone the repo
git clone https://github.com/yourusername/your-repo.git
cd your-repo

# Create virtual environment
python -m venv myenv
source myenv/bin/activate  # Mac/Linux
myenv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Pull Ollama model
ollama pull llama3.2
```

---

## 🔑 Environment Setup

Create a `.env` file in root folder:

```bash
GROQ_API_KEY=your_groq_api_key_here
```

Get your Groq API key from: https://console.groq.com

---

## 📋 Requirements

```txt
langchain
langchain-core
langchain-community
langchain-groq
langchain-ollama
langchain-text-splitters
streamlit
faiss-cpu
chromadb
langchain-objectbox
pypdf
python-dotenv
```

---

## 💡 Key Concepts

### Chunk Size & Overlap
```
chunk_size    = max characters per chunk
chunk_overlap = shared characters between chunks

Rule of thumb:
chunk_size=1000, chunk_overlap=200 (20% overlap)

Small chunk → precise retrieval, less context
Large chunk → more context, less precise
```

### Why `st.session_state`?
```
Streamlit reruns entire script on every interaction.
session_state preserves expensive operations like:
- Loading documents
- Creating embeddings
- Building vector store

Without it → rebuilds everything on every click ❌
With it    → builds once, reuses always ✅
```

## 🙌 Acknowledgements

- [LangChain](https://docs.langchain.com/)
- [Groq](https://console.groq.com/)
- [Ollama](https://ollama.ai/)
- [Streamlit](https://streamlit.io/)
- [FAISS](https://github.com/facebookresearch/faiss)
- [Chroma](https://www.trychroma.com/)
- [ObjectBox](https://objectbox.io/)
