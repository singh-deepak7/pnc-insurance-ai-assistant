# 🧠 PNC Insurance AI Assistant

An **AI-powered multi-agent insurance assistant** that leverages **LLMs, RAG (Retrieval-Augmented Generation), and Hybrid Search (BM25 + Vector Search)** to answer insurance-related queries with high accuracy and contextual understanding.

---

## 🚀 Overview

Insurance documents are complex, lengthy, and difficult to interpret. This project solves that by building an **intelligent AI assistant** that can:

* Understand insurance policies (PDFs)
* Answer user queries in natural language
* Retrieve relevant context using hybrid search
* Use multiple AI agents to improve reasoning and response quality

This system simulates a **real-world enterprise AI architecture** for insurance analytics and customer support.

---

## 🏗️ Architecture

### 🔹 High-Level Flow

```
User Query
   ↓
Planner Agent
   ↓
Researcher Agent(s)
   ↓
Search (Vector DB)
   ↓
Synthesizer Agent
   ↓
Final Response
```

---

## 🤖 Multi-Agent Design

### 1. Planner Agent

* Breaks user query into sub-questions
* Decides what information is needed
* Orchestrates the workflow

### 2. Researcher Agent

* Fetches relevant information from:

  * Vector DB (semantic search) 

### 3. Synthesizer Agent

* Combines all findings
* Generates final coherent response using LLM

### 4. Orchestrator

* Controls execution flow between agents
* Manages retries, aggregation, and final output

---

## 📂 Project Structure

```
pnc-insurance-ai-assistant/
│
|── rag_pipleine/
|   └── cost_tracker.py       # Estimates token usage and embedding cost before running the embedding process.
|   └── hybrid_search.py      # Implements BM25-based keyword search to complement semantic (vector) retrieval. (TODO)
|   └── ingest.py             # End-to-end pipeline that loads PDFs, cleans/chunks data, creates embeddings, and builds the FAISS + BM25 indexes.
|   └── vectore_store.py      # Loads the saved FAISS vector database for use during query-time retrieval.
│
├── agents/
│   ├── planner.py            # Break Query (LLM)
│   ├── researcher.py         # Retrieve docs ( FAISS + embeddings)
│   ├── synthesizer.py        # Generate answer (LLM)
│   └── orchestrator.py       # Control flow (python)
│
├── backend/app
│   └── main.py               # Initializes the FastAPI server and registers API routes.
|   └── routes/ 
|       └── query.py          # Defines the /query API endpoint and formats the response returned to the client.
|   └── services/ 
|       └── rag_service.py    # Core service that handles query processing using RAG and multi-agent orchestration.
|   └── frontend/ 
|       └── app.py # UI
│
├── data/
│   └── raw/                  # Insurance PDFs
|
|── run.sh # script to run the app locally 
|
├── requirements.txt
└── README.md
```

---

## ⚙️ Tech Stack

* **LLM**: OpenAI / LLM (e.g., GPT / Llama)
* **Embeddings**: `text-embedding-3-small`
* **Vector DB**: FAISS
* **Search**: Vector DB
* **Frameworks**:

  * LangChain
  * Python

* **UI**:

  * Gradio

---

## 🔄 Ingestion Pipeline

1. Load PDF documents
2. Split into chunks
3. Generate embeddings
4. Store in FAISS vector DB

---

## 💡 Features

* ✅ Multi-agent architecture (Planner, Researcher, Synthesizer)
* ✅ Search (Vector)
* ✅ Fast semantic retrieval
* ✅ Context-aware answers
* ✅ Scalable design for enterprise use
* ✅ Modular & extensible codebase

---

## ▶️ How to Run

### 1. Clone Repo

```bash
git clone https://github.com/singh-deepak7/ai.git
cd pnc-insurance-ai-assistant
```

### 2. Install Dependencies

```bash
# Install uv (skip if already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create a virtual environment and install dependencies
uv venv .venv --python 3.11 && source .venv/bin/activate
uv pip install -r requirements.txt
```

### 3. Set Environment Variables

```bash
export OPENAI_API_KEY=your_key
```

### 4. Run Ingestion

```bash
python -m rag_pipeline.ingest
```

### 5. Start Application

```bash
bash run.sh  
```

---

## 🧪 Example Queries

* “What does homeowners insurance cover?”
* "What does auto insurance cover?"
* “Explain deductible in simple terms”
* “Compare liability vs comprehensive insurance”
* “What is not covered in this policy?”

---

## 📊 Real-World Relevance

AI assistants like this are transforming insurance by:

* Automating customer support
* Reducing operational costs
* Improving accuracy in policy interpretation
* Enhancing customer experience

---

## 🔮 Future Enhancements

* 🔹 Voice-based assistant
* 🔹 Fine-tuned domain-specific models
* 🔹 Real-time policy updates
* 🔹 Integration with AWS Bedrock
* 🔹 Personalized recommendations

---

## 👨‍💻 Author

Deepak Singh
Senior Lead Developer | AI & Cloud Enthusiast

