from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import os

from backend.vocab import vocab
from backend.gemini_client import GeminiClient
from backend.planner import build_render_plan
from backend.retrieval_service import RetrievalService
from backend.sign_seq import SignSequenceManager
from fastapi import File, UploadFile
import shutil
import tempfile

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount Static Directories
# Mount sgsl_dataset to /static/sgsl_dataset
DATASET_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "sgsl_dataset")
PROCESSED_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "sgsl_processed")

if os.path.exists(DATASET_PATH):
    app.mount("/static/sgsl_dataset", StaticFiles(directory=DATASET_PATH), name="sgsl_dataset")
if os.path.exists(PROCESSED_PATH):
    app.mount("/static/sgsl_processed", StaticFiles(directory=PROCESSED_PATH), name="sgsl_processed")


# Components
gemini = GeminiClient()
retrieval_service = RetrievalService(os.path.join(PROCESSED_PATH, "retrieval_artifacts"))
sign_seq_manager = SignSequenceManager(os.path.join(PROCESSED_PATH, "landmarks_pkl"))

class GlossRequest(BaseModel):
    text: str

class RenderPlanItem(BaseModel):
    token: str
    sign_name: Optional[str]
    type: str
    assets: Dict[str, str]

class TranslateResponse(BaseModel):
    gloss: List[str]
    unmatched: List[str]
    plan: List[Dict[str, Any]]
    notes: Optional[str] = None

@app.get("/health")
def health():
    return {
        "status": "ok", 
        "vocab_size": len(vocab.get_allowed_tokens()),
        "gemini_model": gemini.model.model_name if gemini.model else "Mock Mode"
    }

@app.post("/api/translate", response_model=TranslateResponse)
def translate(req: GlossRequest):
    # 1. Text to Gloss (Gemini)
    print(f"Translating: {req.text}")
    gloss_result = gemini.text_to_gloss(req.text)
    gloss_tokens = gloss_result.get("gloss", [])
    unmatched = gloss_result.get("unmatched", [])
    
    # 2. Gloss to Plan (Planner)
    plan = build_render_plan(gloss_tokens)
    
    return {
        "gloss": gloss_tokens,
        "unmatched": unmatched,
        "plan": plan,
        "notes": gloss_result.get("notes")
    }

@app.post("/api/sign2text")
async def sign_to_text(file: UploadFile = File(...)):
    # Save temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name
        
    try:
        # Search
        results = retrieval_service.search(tmp_path, k=5)
        return {"results": results}
    finally:
        # Cleanup
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

@app.get("/api/pose/{sign_name}")
def get_pose_data(sign_name: str):
    """Get pose data (hand landmarks) for a given sign."""
    try:
        pose_data = sign_seq_manager.get_sign_frames(sign_name)
        if pose_data is None:
            raise HTTPException(status_code=404, detail=f"Pose data not found for sign: {sign_name}")
        return pose_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading pose data: {str(e)}")

