from fastapi import APIRouter, UploadFile, File
from app.rag.pipeline import build_pipeline

router = APIRouter()

@router.post("/index")
async def index_pdf(file: UploadFile = File(...)):

    path = f"data/raw_pdfs/{file.filename}"

    with open(path, "wb") as f:
        f.write(await file.read())

    result = build_pipeline(path)

    return result
