from fastapi import APIRouter
from app.services.rag_service import ask_question

router = APIRouter()

@router.post("/query")
def query(q: str):
    result = ask_question(q)

    return {
        "response": result.get("answer", ""),
        "sources": result.get("sources", []),
        "trace": result.get("trace", []),
        "confidence": float(result.get("confidence", 0.0))  # ✅ force safe type
    }