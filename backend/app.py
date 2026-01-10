from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import os
import requests
import asyncio

from backend.vocab import vocab
from backend.gemini_client import GeminiClient
from backend.planner import build_render_plan

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


class TranscribeRequest(BaseModel):
    audio_data: str  # Base64 encoded audio
    mime_type: str = "audio/webm"


class TranscribeResponse(BaseModel):
    transcription: str


@app.post("/api/transcribe", response_model=TranscribeResponse)
def transcribe_audio(req: TranscribeRequest):
    """
    Transcribe audio to text using Gemini.
    """
    print(f"Received transcription request (audio mime_type: {req.mime_type})")
    
    # Transcribe audio using Gemini
    result = gemini.transcribe_audio(req.audio_data, req.mime_type)
    
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    
    transcription = result.get("transcription", "")
    print(f"Transcription: {transcription}")
    
    return {"transcription": transcription}


@app.get("/api/webrtc/ice-servers")
async def get_ice_servers():
    """
    Get ICE servers (STUN/TURN) for WebRTC connections.
    
    Uses free TURN servers:
    - Open Relay by Metered (requires API key, but provides free 20GB/month)
    - Public test TURN servers as fallback
    
    To use Open Relay with credentials (recommended for production):
    1. Sign up at https://www.metered.ca/tools/openrelay/
    2. Get your API key
    3. Set METERED_TURN_API_KEY environment variable
    """
    ice_servers = [
        # STUN servers (free, no credentials needed)
        {"urls": "stun:stun.l.google.com:19302"},
        {"urls": "stun:stun1.l.google.com:19302"},
        {"urls": "stun:stun2.l.google.com:19302"},
        {"urls": "stun:stun.stunprotocol.org:3478"},
    ]
    
    # Try to get TURN credentials from Metered Open Relay if API key is set
    metered_api_key = os.getenv("METERED_TURN_API_KEY")
    
    if metered_api_key:
        try:
            def fetch_metered_credentials():
                response = requests.get(
                    "https://api.metered.ca/api/v1/turn/credentials",
                    params={"apiKey": metered_api_key},
                    timeout=5.0
                )
                return response
            
            response = await asyncio.to_thread(fetch_metered_credentials)
            if response.status_code == 200:
                data = response.json()
                if "iceServers" in data:
                    # Metered returns full iceServers array, merge with our STUN servers
                    return {"iceServers": data["iceServers"]}
        except Exception as e:
            print(f"Warning: Failed to fetch Metered TURN credentials: {e}")
            print("Falling back to public TURN servers...")
    
    # Fallback: Add TURN servers
    # Note: Most reliable free TURN servers require API keys for security
    # For production, get a free API key from Metered (20GB/month free): https://www.metered.ca/tools/openrelay/
    
    # Open Relay public TURN servers (works with credentials from API key above)
    # If you have METERED_TURN_API_KEY set, the code above will fetch credentials automatically
    # These URLs below are placeholders - actual credentials come from the API
    public_turn_servers = [
        # Note: These may not work without credentials. For best results, get a free Metered API key:
        # 1. Sign up at https://www.metered.ca/tools/openrelay/ (free, no credit card)
        # 2. Get your API key from the dashboard
        # 3. Set environment variable: export METERED_TURN_API_KEY=your_api_key
        # 4. Restart the backend server
        {
            "urls": [
                "turn:openrelay.metered.ca:80",
                "turn:openrelay.metered.ca:443",
                "turn:openrelay.metered.ca:443?transport=tcp"
            ]
            # Note: Credentials are typically required for these servers
            # The API endpoint above will add credentials if METERED_TURN_API_KEY is set
        },
    ]
    
    ice_servers.extend(public_turn_servers)
    
    # Log configuration status
    if metered_api_key:
        print("Using Metered TURN API (if fetch succeeded above)")
    else:
        print("WARNING: METERED_TURN_API_KEY not set. TURN servers may not work without credentials.")
        print("To fix ICE failures, get a free API key at https://www.metered.ca/tools/openrelay/")
        print("Then set: export METERED_TURN_API_KEY=your_api_key")
    
    return {"iceServers": ice_servers}


# Mount frontend LAST (so it doesn't override API routes)
# This serves the frontend at http://127.0.0.1:8000/
if os.path.exists(FRONTEND_PATH):
    app.mount("/", StaticFiles(directory=FRONTEND_PATH, html=True), name="frontend")


