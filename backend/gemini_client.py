import os
import json
import base64
from google import genai as genai_live
from google.genai import types
from typing import List, Dict, Any, Optional
import sys
import asyncio
import io
from pydub import AudioSegment

# Ensure backend can be imported if running as script
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from backend.vocab import vocab
from dotenv import load_dotenv

# Load .env file
load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))

class GeminiClient:
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        self.model = None  # Keep for backward compatibility/health check
        # Initialize client with v1alpha API version for Live API support
        if self.api_key:
            self.client = genai_live.Client(
                api_key=self.api_key,
                http_options=types.HttpOptions(api_version="v1alpha")
            )
        else:
            print("Warning: GEMINI_API_KEY not set. Using mock mode.")
            self.client = None
    
    @property
    def live_client(self):
        """Backward compatibility: live_client is the same as client"""
        return self.client

    def text_to_gloss(self, text: str, allowed_tokens: List[str] = None) -> Dict[str, Any]:
        """
        Translate text to gloss using permitted tokens.
        If allowed_tokens is None, fetches all from vocab.
        Uses google.genai Client API (new API).
        """
        if allowed_tokens is None:
            allowed_tokens = vocab.get_allowed_tokens(text)

        if not self.client:
            return self._mock_response(text, allowed_tokens)

        # Construct prompt
        token_str = ", ".join(allowed_tokens)
        
        prompt = f"""
        You are a Singapore Sign Language (SGSL) translator.
        Translate the following English text into SGSL Gloss.
        
        Constraint: SGSL often uses Subject-Object-Verb or Topic-Comment structure.
        Constraint: You MUST use ONLY words from the provided vocabulary list below.
        If a concept is not in the vocabulary, try to find a synonym in the vocabulary (e.g., "MUM" -> "MOTHER").
        If you cannot translate key parts, put the original word in 'unmatched'.
        
        Vocabulary:
        [{token_str}]
        
        Input Text: "{text}"
        
        Output JSON format strictly:
        {{
          "gloss": ["TOKEN1", "TOKEN2", ...],
          "unmatched": ["word1", ...],
          "notes": "explanation"
        }}
        """
        
        try:
            # Use new google.genai Client API
            # Using gemini-3-flash-preview for better performance, but can use gemini-2.0-flash if preferred
            response = self.client.models.generate_content(
                model="gemini-3-flash-preview",
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    thinking_config=types.ThinkingConfig(thinking_level="low")  # Use low for faster response
                )
            )
            
            # Extract text from response
            # The new google.genai API returns response.text directly
            text_resp = response.text
            
            if not text_resp or not text_resp.strip():
                raise ValueError("Empty response from Gemini API")
            
            # Parse JSON
            data = json.loads(text_resp)
            return self.validate_gloss(data)
        except Exception as e:
            print(f"Gemini Error: {e}")
            import traceback
            traceback.print_exc()
            return {"gloss": [], "unmatched": [], "error": str(e)}

    def validate_gloss(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ensure all tokens in gloss are actually in vocab.
        """
        raw_gloss = data.get("gloss", [])
        validated_gloss = []
        unmatched = data.get("unmatched", [])
        
        for token in raw_gloss:
            # 1. Canonicalize
            canon_token = vocab.canon(token)
            
            # 2. Check alias
            canon_token = vocab.apply_aliases(canon_token)
            
            # 3. Check exist
            if vocab.validate_token(canon_token):
                validated_gloss.append(canon_token)
            else:
                # Fallback: maybe it's a synonym not in alias list?
                # For now, mark as unmatched
                unmatched.append(token)
        
        data["gloss"] = validated_gloss
        data["unmatched"] = unmatched
        return data

    def _mock_response(self, text: str, allowed_tokens: List[str]):
        """Simple keyword matching fallback."""
        words = text.upper().split()
        gloss = []
        unmatched = []
        
        vocab_set = set(allowed_tokens)
        
        for w in words:
            clean = "".join(c for c in w if c.isalnum() or c=='_')
            clean = vocab.apply_aliases(clean)
            
            if clean in vocab_set:
                gloss.append(clean)
            else:
                unmatched.append(w)
                
        return {
            "gloss": gloss,
            "unmatched": unmatched,
            "notes": "Mock response (no API key)"
        }

    def _convert_audio_to_pcm(self, audio_bytes: bytes, mime_type: str) -> bytes:
        """
        Convert audio bytes to PCM format (16-bit, little-endian, 16kHz).
        Supports WebM, WAV, and other formats via pydub.
        """
        try:
            # Extract format from mime_type (e.g., "audio/webm" -> "webm")
            format_str = mime_type.split(';')[0].split('/')[-1]
            if format_str not in ['webm', 'wav', 'mp3', 'ogg', 'm4a', 'flac']:
                # Try to detect format or default to webm
                format_str = 'webm'
            
            # Create AudioSegment from bytes
            audio = AudioSegment.from_file(io.BytesIO(audio_bytes), format=format_str)
            
            # Convert to PCM: 16-bit, mono, 16kHz
            audio = audio.set_frame_rate(16000)
            audio = audio.set_channels(1)
            audio = audio.set_sample_width(2)  # 16-bit = 2 bytes
            
            # Export as raw PCM (little-endian)
            pcm_bytes = audio.raw_data
            return pcm_bytes
            
        except Exception as e:
            print(f"Error converting audio to PCM: {e}")
            raise  # Re-raise to allow fallback handling

    async def transcribe_audio_live(self, audio_base64: str, mime_type: str = "audio/webm") -> Dict[str, Any]:
        """
        Transcribe audio using standard Gemini API.
        This method uses the standard generate_content API for reliable transcription.
        Maintains async interface for compatibility with endpoint.
        """
        # Use the working standard transcription method
        # Run it in a thread pool to maintain async interface
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.transcribe_audio, audio_base64, mime_type)

    def transcribe_audio(self, audio_base64: str, mime_type: str = "audio/webm") -> Dict[str, Any]:
        """
        Transcribe audio to text using Gemini's multimodal capabilities.
        Uses google.genai Client API with gemini-3-flash-preview.
        """
        if not self.client:
            return {
                "transcription": "",
                "error": "No API key - audio transcription requires Gemini API"
            }
        
        try:
            audio_bytes = base64.b64decode(audio_base64)
        except Exception as e:
            return {
                "transcription": "",
                "error": f"Failed to decode audio: {str(e)}"
            }
        
        prompt = """
        You are a highly accurate transcription assistant.
        Listen to the provided audio carefully and transcribe it exactly into English text.
        Do not include any interpretations, just the spoken words.
        
        If no speech is detected, return an empty string for transcription.
        
        Output format: JSON
        {
          "transcription": "The spoken text"
        }
        """
        
        try:
            # Use new google.genai Client API for transcription
            response = self.client.models.generate_content(
                model="gemini-3-flash-preview",
                contents=[
                    types.Content(
                        role="user",
                        parts=[
                            types.Part(text=prompt),
                            types.Part(
                                inline_data=types.Blob(
                                    mime_type=mime_type,
                                    data=audio_bytes
                                )
                            )
                        ]
                    )
                ],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    thinking_config=types.ThinkingConfig(thinking_level="low")
                )
            )
            
            text_resp = response.text
            print(f"Gemini Transcription Response: {text_resp}")
            
            data = json.loads(text_resp)
            return data
            
        except Exception as e:
            print(f"Gemini Audio Error: {e}")
            import traceback
            traceback.print_exc()
            # If JSON generation fails or refusal occurs, try to extract text from raw response
            try:
                if hasattr(response, 'text') and response.text:
                    return {"transcription": response.text.strip()}
            except:
                pass
            return {
                "transcription": "",
                "error": str(e)
            }
