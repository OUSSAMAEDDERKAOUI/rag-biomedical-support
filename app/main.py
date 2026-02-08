from fastapi import FastAPI
from app.api.v1 import auth
from app.api.v1 import rag_router

from app.db.init_db import init_db

app = FastAPI(title="RAG Biomedical Support")

init_db()

app.include_router(auth.router, prefix="/api/v1/auth", tags=["auth"])
app.include_router(rag_router.router, prefix="/api/v1/index", tags=["rag"])

# app.include_router(rag_router.router)
