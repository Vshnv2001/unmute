from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Optional, Dict, Any, Set
import os
import asyncio
import json

from backend.vocab import vocab
from backend.gemini_client import GeminiClient
from backend.planner import build_render_plan
from backend.sign_seq import SignSequenceManager
from backend.gcs_storage import USE_GCS, get_dataset_info

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount Static Directories for sign language assets only (when not using GCS)
# When USE_GCS=true, assets are served directly from Google Cloud Storage
DATASET_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "sgsl_dataset")
PROCESSED_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "sgsl_processed")

if not USE_GCS:
    print("[App] Using local static file serving")
    if os.path.exists(DATASET_PATH):
        app.mount("/static/sgsl_dataset", StaticFiles(directory=DATASET_PATH), name="sgsl_dataset")
    if os.path.exists(PROCESSED_PATH):
        app.mount("/static/sgsl_processed", StaticFiles(directory=PROCESSED_PATH), name="sgsl_processed")
else:
    print(f"[App] Using GCS for static files: {get_dataset_info()['public_url']}")



# Components
gemini = GeminiClient()
sign_mgr = SignSequenceManager()

class GlossRequest(BaseModel):
    text: str
    language: Optional[str] = None  # Language code (e.g., 'en', 'zh', 'ms', 'ta'). If None, auto-detects.

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
    detected_language: Optional[str] = None  # Language code if auto-detected

@app.get("/health")
def health():
    storage_info = get_dataset_info()
    return {
        "status": "ok", 
        "vocab_size": len(vocab.get_allowed_tokens()),
        "storage": storage_info,
    }

@app.post("/api/translate", response_model=TranslateResponse)
def translate(req: GlossRequest):
    # 1. Text to Gloss (Gemini) with language support
    print(f"Translating: {req.text} (language: {req.language or 'auto-detect'})")
    gloss_result = gemini.text_to_gloss(req.text, language=req.language)
    gloss_tokens = gloss_result.get("gloss", [])
    unmatched = gloss_result.get("unmatched", [])
    detected_language = gloss_result.get("detected_language")
    
    # 2. Gloss to Plan (Planner)
    plan = build_render_plan(gloss_tokens)
    
    return {
        "gloss": gloss_tokens,
        "unmatched": unmatched,
        "plan": plan,
        "notes": gloss_result.get("notes"),
        "detected_language": detected_language
    }

@app.get("/api/sign/{sign_name}/landmarks")
def get_landmarks(sign_name: str):
    """Return 3D landmark frames for a sign (both hand and pose data if available)."""
    hand_data = sign_mgr.get_sign_frames(sign_name)
    pose_data = sign_mgr.get_sign_pose_frames(sign_name)
    
    # Save hand data to hand.txt and pose data to pose.txt
    with open("hand.txt", "w") as f:
        json.dump(hand_data, f)
    with open("pose.txt", "w") as f:
        json.dump(pose_data, f)
    
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
    
    # Print response to resp.txt
    with open("resp.txt", "w") as f:
        json.dump(response, f)
    return response


class TranscribeRequest(BaseModel):
    audio_data: str  # Base64 encoded audio
    mime_type: str = "audio/webm"
    language: Optional[str] = None  # Language code (e.g., 'en', 'zh', 'ms', 'ta'). If None, auto-detects.
    auto_translate: bool = False  # If True, automatically translate transcription to sign language


class TranscribeResponse(BaseModel):
    transcription: str
    detected_language: Optional[str] = None  # Language code if auto-detected
    gloss: Optional[List[str]] = None  # Only included if auto_translate is True
    unmatched: Optional[List[str]] = None  # Only included if auto_translate is True
    plan: Optional[List[Dict[str, Any]]] = None  # Only included if auto_translate is True
    notes: Optional[str] = None  # Only included if auto_translate is True


@app.post("/api/transcribe")
async def transcribe_audio(req: TranscribeRequest):
    """
    Transcribe audio to text using Gemini Live API with automatic VAD.
    Supports multiple languages: English, Chinese, Malay, Tamil, and others.
    If auto_translate is True, automatically translates the transcription to sign language.
    
    Args:
        req: TranscribeRequest with audio_data, mime_type, optional language, and auto_translate flag
    """
    print(f"Received transcription request (audio mime_type: {req.mime_type}, language: {req.language or 'auto-detect'}, auto_translate: {req.auto_translate})")
    
    # Transcribe audio using Live API with VAD and language support
    result = await gemini.transcribe_audio_live(req.audio_data, req.mime_type, req.language)
    
    if "error" in result:
        # Fallback to standard transcription method if Live API fails
        print(f"Live API failed, falling back to standard transcription: {result['error']}")
        result = gemini.transcribe_audio(req.audio_data, req.mime_type, req.language)
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
    
    transcription = result.get("transcription", "")
    detected_language = result.get("detected_language")
    print(f"Transcription: {transcription} (detected language: {detected_language})")
    
    # If auto_translate is enabled, automatically translate
    if req.auto_translate and transcription:
        print(f"Auto-translating: {transcription}")
        # Use detected language if available, otherwise use provided language or auto-detect
        translation_lang = detected_language or req.language
        gloss_result = gemini.text_to_gloss(transcription, language=translation_lang)
        gloss_tokens = gloss_result.get("gloss", [])
        unmatched = gloss_result.get("unmatched", [])
        
        # Build render plan
        plan = build_render_plan(gloss_tokens)
        
        # Return full translation response
        return {
            "transcription": transcription,
            "detected_language": detected_language,
            "gloss": gloss_tokens,
            "unmatched": unmatched,
            "plan": plan,
            "notes": gloss_result.get("notes")
        }
    
    # Return just transcription with detected language
    return {
        "transcription": transcription,
        "detected_language": detected_language
    }


# Frontend is now served separately via Vite dev server (unmute-fe)


# ============ WebRTC Signaling Server ============

class ConnectionManager:
    """Manages WebSocket connections and rooms for WebRTC signaling."""
    
    def __init__(self):
        # room_id -> set of WebSocket connections
        self.rooms: Dict[str, Set[WebSocket]] = {}
        # websocket -> (room_id, user_id)
        self.connections: Dict[WebSocket, tuple] = {}
    
    async def join_room(self, websocket: WebSocket, room_id: str, user_id: str):
        await websocket.accept()
        
        if room_id not in self.rooms:
            self.rooms[room_id] = set()
        
        self.rooms[room_id].add(websocket)
        self.connections[websocket] = (room_id, user_id)
        
        # Notify others in the room
        await self.broadcast_to_room(room_id, {
            "type": "user_joined",
            "user_id": user_id,
            "room_id": room_id,
            "user_count": len(self.rooms[room_id])
        }, exclude=websocket)
        
        # Send current users to the new joiner
        await websocket.send_json({
            "type": "room_info",
            "room_id": room_id,
            "user_count": len(self.rooms[room_id])
        })
    
    async def leave_room(self, websocket: WebSocket):
        if websocket not in self.connections:
            return
        
        room_id, user_id = self.connections[websocket]
        
        if room_id in self.rooms:
            self.rooms[room_id].discard(websocket)
            if not self.rooms[room_id]:
                del self.rooms[room_id]
            else:
                await self.broadcast_to_room(room_id, {
                    "type": "user_left",
                    "user_id": user_id,
                    "room_id": room_id
                })
        
        del self.connections[websocket]
    
    async def broadcast_to_room(self, room_id: str, message: dict, exclude: WebSocket = None):
        if room_id not in self.rooms:
            return
        
        for connection in self.rooms[room_id]:
            if connection != exclude:
                try:
                    await connection.send_json(message)
                except:
                    pass
    
    async def relay_message(self, websocket: WebSocket, message: dict):
        """Relay signaling messages (offer, answer, ice-candidate) to peers."""
        if websocket not in self.connections:
            return
        
        room_id, sender_id = self.connections[websocket]
        message["sender_id"] = sender_id
        
        # If target_id specified, send only to that user
        target_id = message.get("target_id")
        if target_id:
            for conn, (r_id, u_id) in self.connections.items():
                if r_id == room_id and u_id == target_id:
                    await conn.send_json(message)
                    return
        
        # Otherwise broadcast to all in room
        await self.broadcast_to_room(room_id, message, exclude=websocket)


manager = ConnectionManager()


@app.websocket("/ws/room/{room_id}/{user_id}")
async def websocket_endpoint(websocket: WebSocket, room_id: str, user_id: str):
    """WebSocket endpoint for WebRTC signaling."""
    print(f"[WebRTC] User {user_id} joining room {room_id}")
    await manager.join_room(websocket, room_id, user_id)
    
    try:
        while True:
            data = await websocket.receive_json()
            msg_type = data.get("type")
            print(f"[WebRTC] Received {msg_type} from {user_id}")
            
            if msg_type in ["offer", "answer", "ice-candidate"]:
                # Relay WebRTC signaling messages
                await manager.relay_message(websocket, data)
            elif msg_type == "chat":
                # Optional: relay chat messages
                await manager.relay_message(websocket, data)
            elif msg_type == "sign-translation":
                # Relay sign language translation to all other users in room
                print(f"[WebRTC] Relaying sign translation from {user_id}")
                await manager.relay_message(websocket, data)
    
    except WebSocketDisconnect:
        print(f"[WebRTC] User {user_id} disconnected from room {room_id}")
        await manager.leave_room(websocket)
    except Exception as e:
        print(f"[WebRTC] WebSocket error: {e}")
        await manager.leave_room(websocket)
