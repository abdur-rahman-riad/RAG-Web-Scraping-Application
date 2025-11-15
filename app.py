import streamlit as st
import os

# LangChain Imports
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


# -------------------------
# Streamlit Page Settings
# -------------------------
st.set_page_config(page_title="Web Scraping RAG App", layout="wide")
st.title("🌐 Web Scraping RAG Assistant")
st.caption("Enter a website URL → Build VectorDB → Ask Questions Using RAG")


# -------------------------
# Sidebar (API Key)
# -------------------------
with st.sidebar:
    st.header("🔐 API Settings")

    st.write("Add your Gemini API key below or store it in Streamlit Secrets.")
    api_key = st.text_input("Gemini API Key", type="password")

    if not api_key:
        st.warning("Please enter your Gemini API key to enable the LLM.")


# -------------------------
# Website Input Section
# -------------------------
st.subheader("🔗 Enter Website URL")

url = st.text_input("Website URL", placeholder="https://example.com")


# -----------------------------------
# Scrape Website → Build Vector Store
# -----------------------------------
if st.button("🚀 Scrape & Build Vector Database"):
    if not url.strip():
        st.error("❌ Please enter a valid website URL.")
        st.stop()

    if not api_key:
        st.error("❌ Enter your Gemini API key first.")
        st.stop()

    with st.spinner("🔍 Scraping website content..."):
        loader = WebBaseLoader(url)
        docs = loader.load()

    st.success(f"✅ Loaded {len(docs)} document(s) from the website.")

    # Step 2: Split the documents
    with st.spinner("✂ Splitting documents into chunks..."):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        chunks = splitter.split_documents(docs)

    st.success(f"✅ Created {len(chunks)} chunks.")

    # Step 3: Create embeddings + vector DB
    with st.spinner("⚡ Creating vector database..."):
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        vectordb = Chroma(
            collection_name="web_rag_collection",
            embedding_function=embeddings,
            persist_directory="./web_rag_db"
        )

        vectordb.add_documents(chunks)

    st.success("🎉 Vector database built successfully!")

    # Store in session
    st.session_state["retriever"] = vectordb.as_retriever()

    # LLM setup
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=api_key
    )

    # RAG prompt
    prompt = ChatPromptTemplate.from_template("""
        You are an AI assistant answering questions based ONLY on the website content.

        <context>
        {context}
        </context>

        Question: {question}

        If the answer is not found in the context, reply:
        "The information is not available in the provided website data."

        Answer:
    """)

    rag_chain = (
        RunnableParallel({
            "context": st.session_state["retriever"],
            "question": RunnablePassthrough()
        })
        | prompt
        | llm
        | StrOutputParser()
    )

    st.session_state["rag_chain"] = rag_chain

    st.success("🤖 RAG system is ready! Scroll down to ask questions.")


# -----------------------------------
# Question-Answer Section (Chat UI)
# -----------------------------------
st.subheader("💬 Ask Questions About the Website")

if "rag_chain" not in st.session_state:
    st.info("ℹ️ Scrape a website first to enable RAG.")
else:
    user_query = st.text_input("Ask a question:")

    if user_query:
        with st.spinner("⏳ Thinking..."):
            response = st.session_state["rag_chain"].invoke(user_query)

        st.markdown("### 🧠 Answer")
        st.write(response)
