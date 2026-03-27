from langchain_community.vectorstores import FAISS
#from langchain_ollama import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DB_PATH = os.path.join(BASE_DIR, "data/vector_db")

def load_vectorstore():
    #embeddings = OllamaEmbeddings(model="llama3.2:3b")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    if os.path.exists(DB_PATH):
        return FAISS.load_local(
            DB_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )

    raise Exception("Vector DB not found. Run ingest.py first.")