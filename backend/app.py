from fastapi import FastAPI, HTTPException, Depends, status, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.sessions import SessionMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import os
import secrets
import json
from sqlalchemy.orm import Session

from backend.vocab import vocab
from backend.gemini_client import GeminiClient
from backend.planner import build_render_plan
from backend.models import Room, SessionLocal, init_db

# Initialize Database
init_db()

app = FastAPI()

# Add Session Middleware (still useful for general state if needed)
app.add_middleware(SessionMiddleware, secret_key=os.environ.get("SESSION_SECRET", "another_secret_key"))

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# WebSocket Connection Manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, List[WebSocket]] = {}

    async def connect(self, room_id: str, websocket: WebSocket):
        await websocket.accept()
        if room_id not in self.active_connections:
            self.active_connections[room_id] = []
        self.active_connections[room_id].append(websocket)

    def disconnect(self, room_id: str, websocket: WebSocket):
        if room_id in self.active_connections:
            if websocket in self.active_connections[room_id]:
                self.active_connections[room_id].remove(websocket)
            if not self.active_connections[room_id]:
                del self.active_connections[room_id]

    async def broadcast(self, room_id: str, message: dict):
        if room_id in self.active_connections:
            for connection in self.active_connections[room_id]:
                try:
                    await connection.send_json(message)
                except:
                    pass

manager = ConnectionManager()

# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# --- Meeting Rooms (Anonymous) ---

@app.post("/api/rooms/create")
def create_room(db: Session = Depends(get_db)):
    room_id = secrets.token_urlsafe(8)
    new_room = Room(room_id=room_id)  # No created_by
    db.add(new_room)
    db.commit()
    return {"room_id": room_id}

@app.get("/api/rooms/join/{room_id}")
def join_room(room_id: str, db: Session = Depends(get_db)):
    room = db.query(Room).filter(Room.room_id == room_id).first()
    if not room:
        raise HTTPException(status_code=404, detail="Room not found")
    return {"message": "Room exists", "room_id": room_id}

@app.websocket("/ws/meeting/{room_id}")
async def websocket_meeting(websocket: WebSocket, room_id: str):
    await manager.connect(room_id, websocket)
    try:
        while True:
            data = await websocket.receive_json()
            await manager.broadcast(room_id, data)
    except WebSocketDisconnect:
        manager.disconnect(room_id, websocket)

# --- Original Translation Logic ---

DATASET_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "sgsl_dataset")
PROCESSED_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "sgsl_processed")
FRONTEND_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend")

if os.path.exists(DATASET_PATH):
    app.mount("/static/sgsl_dataset", StaticFiles(directory=DATASET_PATH), name="sgsl_dataset")
if os.path.exists(PROCESSED_PATH):
    app.mount("/static/sgsl_processed", StaticFiles(directory=PROCESSED_PATH), name="sgsl_processed")

gemini = GeminiClient()

class GlossRequest(BaseModel):
    text: str

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
    gloss_result = gemini.text_to_gloss(req.text)
    gloss_tokens = gloss_result.get("gloss", [])
    unmatched = gloss_result.get("unmatched", [])
    plan = build_render_plan(gloss_tokens)
    return {
        "gloss": gloss_tokens,
        "unmatched": unmatched,
        "plan": plan,
        "notes": gloss_result.get("notes")
    }

class TranscribeRequest(BaseModel):
    audio_data: str
    mime_type: str = "audio/webm"

class TranscribeResponse(BaseModel):
    transcription: str

@app.post("/api/transcribe", response_model=TranscribeResponse)
def transcribe_audio(req: TranscribeRequest):
    result = gemini.transcribe_audio(req.audio_data, req.mime_type)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return {"transcription": result.get("transcription", "")}

if os.path.exists(FRONTEND_PATH):
    app.mount("/", StaticFiles(directory=FRONTEND_PATH, html=True), name="frontend")
