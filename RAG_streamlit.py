import os
import streamlit as st
from dotenv import load_dotenv

# LangChain imports
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.embeddings import HuggingFaceEmbeddings

# ================== CONFIG ==================

load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

st.title("🩺 Moringa Medical Assistant (RAG)")
st.write("Ask anything about Moringa Tea (from PDF only)")

# ================== LOAD + CACHE ==================
@st.cache_resource
def load_rag_pipeline():
    
    # Load PDF
    loader = PyPDFLoader(r"PDF PATH")
    docs = loader.load()

    # Split
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = splitter.split_documents(docs)

    # Embeddings
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Vector DB
    vectorstore = Chroma.from_documents(documents=splits, embedding=embedding_model)
    retriever = vectorstore.as_retriever()

    # Prompt
    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(
            """You are a medical doctor assistant.
Answer strictly using the provided context only.
Do NOT use prior knowledge.
you can answer only welcome questions.

If the answer is not in the context, say:
"I don't know based on the provided context."

Give clear and professional medical answers."""
        ),
        HumanMessagePromptTemplate.from_template(
            """Context:
{context}

Question:
{question}
"""
        )
    ])

    # LLM
    llm = ChatGroq(model="openai/gpt-oss-120b")

    # Parser
    parser = StrOutputParser()

    # RAG chain
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | parser
    )

    return rag_chain

rag_chain = load_rag_pipeline()

# ================== CHAT MEMORY ==================
if "messages" not in st.session_state:
    st.session_state.messages = []

# Show old messages
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).markdown(msg["content"])

# ================== USER INPUT ==================
user_input = st.chat_input("Ask your medical question...")

if user_input:
    st.chat_message("user").markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    # STREAM RESPONSE
    with st.chat_message("ai"):
        response = st.write_stream(rag_chain.stream(user_input))

    # Save response
    st.session_state.messages.append({"role": "ai", "content": response})