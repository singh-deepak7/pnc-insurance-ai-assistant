from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from rag_pipeline.vector_store import load_vectorstore
from agents.orchestrator import run_multi_agent


# Load vector DB
vectorstore = load_vectorstore()
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# LLM (optimized)
llm = Ollama(
    model="llama3.2:3b",
    num_predict=200,
    temperature=0.3
)

# Prompt (short + controlled)
prompt = ChatPromptTemplate.from_template("""
You are an insurance assistant.

Answer briefly (3-5 sentences max) using only the context.

Context:
{context}

Question:
{question}

Answer:
""")

# Format docs (trimmed)
def format_docs(docs):
    return "\n\n".join([
        doc.page_content[:800]
        for doc in docs
    ])

# Chain
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
)

def ask_question(query: str):
    print(f"🔍 Query: {query}")
    #result = rag_chain.invoke(query)
    result = run_multi_agent(query)
    print("✅ Response generated")
    return result