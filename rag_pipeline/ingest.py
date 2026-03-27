import os
import re
from typing import List

import sys

from langchain_community.document_loaders import PyPDFLoader, PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
#from langchain_ollama import OllamaEmbeddings
#from langchain_community.embeddings import OllamaEmbeddings
#from langchain_ollama import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings


# Optional hybrid search
from .hybrid_search import build_bm25

# Cost tracking
from rag_pipeline.cost_tracker import calculate_cost

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

DATA_PATH = "data/raw"
DB_PATH = "data/vector_db"


# ---------------------------
# 🧹 Text Cleaning
# ---------------------------
def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)  # remove extra whitespace
    text = re.sub(r"\n+", "\n", text)
    return text.strip()


# ---------------------------
# 📄 Detect Document Type
# ---------------------------
def detect_doc_type(filename: str) -> str:
    name = filename.lower()

    if "policy" in name:
        return "policy"
    elif "claim" in name:
        return "claims"
    elif "underwriting" in name:
        return "underwriting"
    elif "auto" in name:
        return "auto"
    elif "homeowners" in name:
        return "homeowners"
    elif "report" in name or "sigma" in name:
        return "report"
    else:
        return "general"


# ---------------------------
# 📥 Safe PDF Loader
# ---------------------------
def load_pdf(file_path: str):
    try:
        loader = PyMuPDFLoader(file_path)  # preferred
        return loader.load()
    except Exception:
        try:
            loader = PyPDFLoader(file_path)
            return loader.load()
        except Exception as e:
            print(f"⚠️ Skipping {file_path}: {e}")
            return []


# ---------------------------
# ✂️ Smart Chunking
# ---------------------------
def chunk_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,         # larger chunks for context
        chunk_overlap=100,      # overlap for continuity
        separators=["\n\n", "\n", ".", " "]
    )

    return splitter.split_documents(docs)


# ---------------------------
# 🧠 Enrich Metadata
# ---------------------------
def enrich_metadata(docs, filename):
    doc_type = detect_doc_type(filename)

    for d in docs:
        d.page_content = clean_text(d.page_content)

        d.metadata.update({
            "source": filename,
            "doc_type": doc_type,
            "page": d.metadata.get("page", 0)
        })

    return docs


# ---------------------------
# 🚀 Main Ingestion Pipeline
# ---------------------------
def ingest():
    all_docs = []

    print("📥 Loading PDFs...")

    for file in os.listdir(DATA_PATH):
        if file.endswith(".pdf"):
            path = os.path.join(DATA_PATH, file)

            docs = load_pdf(path)
            docs = enrich_metadata(docs, file)

            all_docs.extend(docs)

            print(f"✅ Loaded: {file} ({len(docs)} pages)")

    if not all_docs:
        raise Exception("❌ No documents loaded. Check your data folder.")

    print("✂️ Chunking documents...")
    chunks = chunk_documents(all_docs)

    # ---------------------------
    # 💰 Cost Estimation
    # ---------------------------
    print("💰 Estimating embedding cost...")

    stats = calculate_cost(chunks)

    print("\n📊 Embedding Stats:")
    print(f"- Total chunks: {stats['total_chunks']}")
    print(f"- Estimated tokens: {stats['total_tokens']:,}")
    print(f"- Avg tokens/chunk: {stats['avg_tokens_per_chunk']}")
    print(f"- Estimated cost: ${stats['estimated_cost']}\n")

    print(f"📊 Total chunks: {len(chunks)}")

    # ---------------------------
    # 🧠 Embeddings
    # ---------------------------
    print("🧠 Creating embeddings...")
    #embeddings = OllamaEmbeddings(model="nomic-embed-text")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # ---------------------------
    # 📦 Vector Store
    # ---------------------------
    print("📦 Building FAISS index...")
    #vectorstore = FAISS.from_documents(chunks, embeddings)
    texts = [c.page_content for c in chunks]
    metadatas = [c.metadata for c in chunks]

    vectorstore = FAISS.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas
    )
    vectorstore.save_local(DB_PATH)

    # ---------------------------
    # 🔍 Hybrid Search (BM25)
    # ---------------------------
    print("🔍 Building BM25 index...")
    bm25 = build_bm25(chunks)

    print("✅ Ingestion complete!")
    print(f"📁 Vector DB saved at: {DB_PATH}")


if __name__ == "__main__":
    ingest()