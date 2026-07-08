from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.config import DEMO_DIR, ROOT_DIR
from app.routers import datasets, evaluation, inference, reports, training

app = FastAPI(title="Materials VLM Post Training Studio", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if DEMO_DIR.exists():
    app.mount("/demo", StaticFiles(directory=DEMO_DIR), name="demo")


@app.get("/health")
def health():
    return {"status": "ok", "project_root": str(ROOT_DIR), "mode": "mock"}


app.include_router(datasets.router)
app.include_router(training.router)
app.include_router(evaluation.router)
app.include_router(inference.router)
app.include_router(reports.router)

