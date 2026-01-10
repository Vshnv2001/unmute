import os
import json
import google.generativeai as genai
from typing import List, Dict, Any
import sys

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
        self.model = None
        if self.api_key:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel('gemini-2.0-flash')
        else:
            print("Warning: GEMINI_API_KEY not set. Using mock mode.")

    def text_to_gloss(self, text: str, allowed_tokens: List[str] = None) -> Dict[str, Any]:
        """
        Translate text to gloss using permitted tokens.
        If allowed_tokens is None, fetches all from vocab.
        """
        if allowed_tokens is None:
            allowed_tokens = vocab.get_allowed_tokens(text)

        if not self.model:
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
            response = self.model.generate_content(prompt, generation_config={"response_mime_type": "application/json"})
            text_resp = response.text
            # Parse JSON
            data = json.loads(text_resp)
            return self.validate_gloss(data)
        except Exception as e:
            print(f"Gemini Error: {e}")
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
