from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import os
import asyncio

from backend.vocab import vocab
from backend.gemini_client import GeminiClient
from backend.planner import build_render_plan
from backend.sign_seq import SignSequenceManager

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
FRONTEND_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend")

if os.path.exists(DATASET_PATH):
    app.mount("/static/sgsl_dataset", StaticFiles(directory=DATASET_PATH), name="sgsl_dataset")
if os.path.exists(PROCESSED_PATH):
    app.mount("/static/sgsl_processed", StaticFiles(directory=PROCESSED_PATH), name="sgsl_processed")



# Components
gemini = GeminiClient()
sign_mgr = SignSequenceManager()

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

@app.get("/api/sign/{sign_name}/landmarks")
def get_landmarks(sign_name: str):
    """Return 3D landmark frames for a sign (both hand and pose data if available)."""
    hand_data = sign_mgr.get_sign_frames(sign_name)
    pose_data = sign_mgr.get_sign_pose_frames(sign_name)
    
    if not hand_data and not pose_data:
        raise HTTPException(status_code=404, detail="Sign data not found")
    
    response = {}
    
    if hand_data:
        response["hand_frames"] = hand_data.get("frames", [])
        response["L_orig"] = hand_data.get("L_orig")
        response["L_max"] = hand_data.get("L_max")
    else:
        response["hand_frames"] = None
    
    if pose_data:
        response["pose_frames"] = pose_data.get("frames", [])
        # Use pose data for L_orig and L_max if hand data is not available
        if not hand_data:
            response["L_orig"] = pose_data.get("L_orig")
            response["L_max"] = pose_data.get("L_max")
    else:
        response["pose_frames"] = None
    
    return response


class TranscribeRequest(BaseModel):
    audio_data: str  # Base64 encoded audio
    mime_type: str = "audio/webm"
    auto_translate: bool = False  # If True, automatically translate transcription to sign language


class TranscribeResponse(BaseModel):
    transcription: str


@app.post("/api/transcribe")
async def transcribe_audio(req: TranscribeRequest):
    """
    Transcribe audio to text using Gemini Live API with automatic VAD.
    If auto_translate is True, automatically translates the transcription to sign language.
    """
    print(f"Received transcription request (audio mime_type: {req.mime_type}, auto_translate: {req.auto_translate})")
    
    # Transcribe audio using Live API with VAD
    result = await gemini.transcribe_audio_live(req.audio_data, req.mime_type)
    
    if "error" in result:
        # Fallback to old method if Live API fails
        print(f"Live API failed, falling back to standard transcription: {result['error']}")
        result = gemini.transcribe_audio(req.audio_data, req.mime_type)
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
    
    transcription = result.get("transcription", "")
    print(f"Transcription: {transcription}")
    
    # If auto_translate is enabled, automatically translate
    if req.auto_translate and transcription:
        print(f"Auto-translating: {transcription}")
        gloss_result = gemini.text_to_gloss(transcription)
        gloss_tokens = gloss_result.get("gloss", [])
        unmatched = gloss_result.get("unmatched", [])
        
        # Build render plan
        plan = build_render_plan(gloss_tokens)
        
        # Return full translation response
        return {
            "transcription": transcription,
            "gloss": gloss_tokens,
            "unmatched": unmatched,
            "plan": plan,
            "notes": gloss_result.get("notes")
        }
    
    # Return just transcription
    return {"transcription": transcription}


# Mount frontend LAST (so it doesn't override API routes)
# This serves the frontend at http://127.0.0.1:8000/
if os.path.exists(FRONTEND_PATH):
    app.mount("/", StaticFiles(directory=FRONTEND_PATH, html=True), name="frontend")


