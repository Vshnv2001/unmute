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

    def text_to_gloss(self, text: str, allowed_tokens: List[str] = None, language: Optional[str] = None) -> Dict[str, Any]:
        """
        Translate text to gloss using permitted tokens.
        Supports multiple input languages: English, Chinese (Simplified/Traditional), 
        Malay, Tamil, and others. Auto-detects language if not specified.
        
        If allowed_tokens is None, fetches all from vocab.
        Uses google.genai Client API (new API).
        
        Args:
            text: Input text in any supported language
            allowed_tokens: List of allowed vocabulary tokens. If None, auto-fetches.
            language: Optional language code (e.g., 'en', 'zh', 'ms', 'ta'). 
                     If None, language is auto-detected.
        """
        if allowed_tokens is None:
            allowed_tokens = vocab.get_allowed_tokens(text)

        if not self.client:
            return self._mock_response(text, allowed_tokens)

        # Construct prompt
        token_str = ", ".join(allowed_tokens)
        
        # Language-specific instructions
        language_instructions = ""
        if language:
            lang_map = {
                'en': 'English',
                'zh': 'Chinese (Simplified or Traditional)',
                'zh-CN': 'Chinese (Simplified)',
                'zh-TW': 'Chinese (Traditional)',
                'ms': 'Malay',
                'ta': 'Tamil',
                'hi': 'Hindi',
                'es': 'Spanish',
                'fr': 'French',
                'de': 'German',
                'ja': 'Japanese',
                'ko': 'Korean'
            }
            lang_name = lang_map.get(language.lower(), language)
            language_instructions = f"\n        Input Language: {lang_name}. Translate from {lang_name} to SGSL Gloss."
        else:
            language_instructions = """
        First, detect the input language automatically. The input may be in:
        - English
        - Chinese (Simplified or Traditional)
        - Malay
        - Tamil
        - Hindi
        - Other languages
        
        Then translate from the detected language to SGSL Gloss."""
        
        prompt = f"""
        You are a multilingual Singapore Sign Language (SGSL) translator.
        Your task is to translate text from ANY language into SGSL Gloss tokens.
        {language_instructions}
        
        Important Constraints:
        1. SGSL often uses Subject-Object-Verb (SOV) or Topic-Comment structure, different from English SVO.
        2. You MUST use ONLY words from the provided vocabulary list below.
        3. For words not in vocabulary, try synonyms (e.g., "MUM" -> "MOTHER", "Mama" -> "MOTHER").
        4. Consider cultural context - SGSL reflects Singapore's multilingual environment.
        5. For Chinese input: Consider tone and context; map to appropriate SGSL concepts.
        6. For Malay/Tamil input: Translate meaningfully, not word-by-word.
        7. If key concepts cannot be translated, include them in 'unmatched' array.
        8. Preserve the semantic meaning and intent of the original text.
        
        Vocabulary (use ONLY these tokens):
        [{token_str}]
        
        Input Text: "{text}"
        
        Output JSON format strictly (no markdown, no code blocks):
        {{
          "gloss": ["TOKEN1", "TOKEN2", ...],
          "unmatched": ["word1", ...],
          "notes": "Brief explanation of translation choices and detected language",
          "detected_language": "language code if auto-detected"
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

    async def transcribe_audio_live(self, audio_base64: str, mime_type: str = "audio/webm", language: Optional[str] = None) -> Dict[str, Any]:
        """
        Transcribe audio using standard Gemini API.
        This method uses the standard generate_content API for reliable transcription.
        Maintains async interface for compatibility with endpoint.
        
        Args:
            audio_base64: Base64 encoded audio data
            mime_type: MIME type of the audio (default: "audio/webm")
            language: Optional language code for transcription
        """
        # Use the working standard transcription method
        # Run it in a thread pool to maintain async interface
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.transcribe_audio, audio_base64, mime_type, language)

    def transcribe_audio(self, audio_base64: str, mime_type: str = "audio/webm", language: Optional[str] = None) -> Dict[str, Any]:
        """
        Transcribe audio to text using Gemini's multimodal capabilities.
        Supports multiple languages: English, Chinese (Simplified/Traditional), 
        Malay, Tamil, Hindi, and others. Auto-detects language if not specified.
        
        Uses google.genai Client API with gemini-3-flash-preview.
        
        Args:
            audio_base64: Base64 encoded audio data
            mime_type: MIME type of the audio (default: "audio/webm")
            language: Optional language code (e.g., 'en', 'zh', 'ms', 'ta').
                     If None, language is auto-detected from audio.
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
        
        # Language-specific instructions
        language_instructions = ""
        if language:
            lang_map = {
                'en': 'English',
                'zh': 'Chinese (Simplified or Traditional)',
                'zh-CN': 'Chinese (Simplified)',
                'zh-TW': 'Chinese (Traditional)',
                'ms': 'Malay',
                'ta': 'Tamil',
                'hi': 'Hindi',
                'es': 'Spanish',
                'fr': 'French',
                'de': 'German',
                'ja': 'Japanese',
                'ko': 'Korean'
            }
            lang_name = lang_map.get(language.lower(), language)
            language_instructions = f"Transcribe the audio in {lang_name}. Output the transcription in the original language (do not translate to English)."
        else:
            language_instructions = """Automatically detect the spoken language from the audio. The audio may contain:
        - English
        - Chinese (Simplified or Traditional)
        - Malay
        - Tamil
        - Hindi
        - Other languages
        
        Transcribe exactly as spoken in the original language. Do not translate to English - preserve the original language of the transcription."""
        
        prompt = f"""
        You are a highly accurate multilingual transcription assistant.
        Listen to the provided audio carefully and transcribe it exactly as spoken.
        
        {language_instructions}
        
        Important guidelines:
        1. Transcribe exactly what is spoken - do not add interpretations or corrections.
        2. Preserve the original language of the speech.
        3. Include punctuation and capitalization as appropriate.
        4. For Chinese transcriptions, use appropriate characters (Simplified or Traditional based on what was spoken).
        5. If multiple languages are spoken, transcribe each in its original language.
        6. If no speech is detected, return an empty string for transcription.
        
        Output format: JSON (no markdown, no code blocks)
        {{
          "transcription": "The spoken text in original language",
          "detected_language": "language code if auto-detected (e.g., 'en', 'zh', 'ms', 'ta')"
        }}
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
