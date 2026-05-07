# 🩺 Moringa Medical Assistant (RAG)

A conversational AI medical assistant built with **Retrieval-Augmented Generation (RAG)** that answers questions strictly from a provided PDF document — in this case, a medical document about **Moringa Tea**. Powered by LangChain, Groq LLM, and deployed as a Streamlit chat app.

---

## 📋 Table of Contents

- [Overview](#overview)
- [How It Works](#how-it-works)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Technologies Used](#technologies-used)

---

## Overview

The Moringa Medical Assistant uses RAG to ensure the model **only answers from the content of your PDF** — not from general prior knowledge. This makes it suitable for domain-specific medical Q&A where accuracy and source-grounding are critical.

Key behaviors:
- Answers questions **only** from the provided PDF context
- Responds with `"I don't know based on the provided context."` if the answer isn't in the document
- Maintains **chat history** within the session
- **Streams** responses in real time

---

## How It Works

```
User Question
      │
      ▼
 Retriever (ChromaDB)
      │  searches embedded PDF chunks
      ▼
 Relevant Context
      │
      ▼
 Prompt (System + Context + Question)
      │
      ▼
 Groq LLM (GPT-OSS 120B)
      │
      ▼
 Streamed Answer → Streamlit Chat UI
```

1. **PDF Loading** — The PDF is loaded using `PyPDFLoader`
2. **Chunking** — Split into chunks of 1000 characters with 200-character overlap
3. **Embeddings** — Each chunk is embedded using `sentence-transformers/all-MiniLM-L6-v2`
4. **Vector Store** — Chunks are stored in a **ChromaDB** vector database
5. **Retrieval** — On each question, the most relevant chunks are retrieved
6. **Generation** — The Groq LLM generates a grounded answer using the retrieved context

The pipeline is cached with `@st.cache_resource` so the PDF is only processed once per session.

---

## Project Structure

```
moringa-rag/
├── RAG_streamlit.py       # Main Streamlit app
├── your_document.pdf      # Medical PDF document (add your own)
├── .env                   # API keys (not committed)
├── requirements.txt       # Dependencies
└── README.md
```

---

## Installation

### Prerequisites

- Python 3.9+
- A [Groq API key](https://console.groq.com/)
- A [LangChain API key](https://smith.langchain.com/) (for tracing, optional)

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/your-username/moringa-rag.git
cd moringa-rag

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set up environment variables
cp .env.example .env
# Then fill in your API keys in .env
```

### requirements.txt

```
streamlit
langchain
langchain-community
langchain-groq
langchain-core
langchain-text-splitters
chromadb
sentence-transformers
pypdf
python-dotenv
```

---

## Configuration

Create a `.env` file in the root directory:

```env
GROQ_API_KEY=your_groq_api_key_here
LANGCHAIN_API_KEY=your_langchain_api_key_here   # optional, for tracing
```

Then update the PDF path in `RAG_streamlit.py`:

```python
loader = PyPDFLoader(r"path/to/your/document.pdf")
```

---

## Usage

```bash
streamlit run RAG_streamlit.py
```

Then open your browser at `http://localhost:8501` and start asking questions about the content of your PDF.

**Example questions:**
- *"What are the health benefits of Moringa Tea?"*
- *"Are there any side effects of consuming Moringa?"*
- *"What does the document say about Moringa and blood pressure?"*

---

## Technologies Used

| Tool | Purpose |
|------|---------|
| [Streamlit](https://streamlit.io/) | Chat web UI |
| [LangChain](https://www.langchain.com/) | RAG pipeline orchestration |
| [Groq](https://groq.com/) | LLM inference (GPT-OSS 120B) |
| [ChromaDB](https://www.trychroma.com/) | Vector store |
| [HuggingFace Sentence Transformers](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) | Text embeddings |
| [PyPDF](https://pypdf.readthedocs.io/) | PDF loading |
| [python-dotenv](https://pypi.org/project/python-dotenv/) | Environment variable management |

---

## Notes

- The assistant is intentionally restricted to the PDF content — it will not answer from general knowledge
- Chat history is maintained within the Streamlit session but resets on page refresh
- The RAG pipeline is cached after the first load for performance

---
