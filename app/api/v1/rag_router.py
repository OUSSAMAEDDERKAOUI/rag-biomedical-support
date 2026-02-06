from fastapi import APIRouter, UploadFile, File
from app.rag.pipeline import build_pipeline

router = APIRouter()

@router.post("/")
async def index_pdf(file: UploadFile = File(...)):

    path = f"data/raw_pdfs/{file.filename}"

    with open(path, "wb") as f:
        f.write(await file.read())

    result = build_pipeline(path)

    return result
from fastapi import APIRouter
from app.services.rag_service import ask_question

router = APIRouter()

@router.post("/ask")
async def ask(data: dict):

    question = data.get("question")

    result = ask_question(question)

    return result
