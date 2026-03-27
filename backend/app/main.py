from fastapi import FastAPI
from app.routes.query import router as query_router

app = FastAPI(title="P&C Insurance AI Assistant")

app.include_router(query_router)

@app.get("/")
def root():
    return {"message": "Insurance AI Assistant Running"}