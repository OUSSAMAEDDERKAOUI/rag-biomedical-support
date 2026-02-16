from fastapi import FastAPI
from app.api.v1 import auth
from app.api.v1 import rag_router
from app.db.init_db import init_db


from fastapi import FastAPI, Response, HTTPException
from prometheus_client import Counter, Histogram, generate_latest
from fastapi import FastAPI, Request


app = FastAPI(title="RAG Biomedical Support")




REQUEST_COUNT = Counter("request_count_total", "Total API requests")
REQUEST_LATENCY = Histogram("request_latency_seconds", "Request latency in seconds")


@app.get("/metrics")
async def get_metrics():
    return Response(generate_latest(), media_type="text/plain")



@app.on_event("startup")
def startup():
    init_db()


app.include_router(auth.router, prefix="/api/v1/auth", tags=["auth"])
app.include_router(rag_router.router, prefix="/api/v1/index", tags=["rag"])




