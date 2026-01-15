import json
import os
from typing import Dict, List, Optional, Set

from backend.gcs_storage import read_json, USE_GCS

# Paths
current_dir = os.path.dirname(os.path.abspath(__file__))
# Assuming sgsl_processed is at the project root, one level up from backend
PROJECT_ROOT = os.path.dirname(current_dir)
VOCAB_PATH = os.path.join(PROJECT_ROOT, "sgsl_processed", "vocab.json")
ALIASES_PATH = os.path.join(current_dir, "aliases.json")

# GCS path for vocab
GCS_VOCAB_PATH = "sgsl_processed/vocab.json"

class VocabLoader:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(VocabLoader, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self.token_to_sign: Dict[str, str] = {}
        self.sign_to_token: Dict[str, str] = {}
        self.aliases: Dict[str, str] = {}
        self.allowed_tokens_list: List[str] = []
        self._load_data()

    def _load_data(self):
        # Load Vocab (from GCS or local)
        data = None
        if USE_GCS:
            print(f"[Vocab] Loading from GCS: {GCS_VOCAB_PATH}")
            data = read_json(GCS_VOCAB_PATH)
        elif os.path.exists(VOCAB_PATH):
            print(f"[Vocab] Loading from local: {VOCAB_PATH}")
            with open(VOCAB_PATH, 'r', encoding='utf-8') as f:
                data = json.load(f)
        
        if data:
            # vocab.json structure is expected to be { "token_to_sign": { ... } }
            if "token_to_sign" in data:
                self.token_to_sign = data["token_to_sign"]
            else:
                # Fallback if the JSON is just the dict directly
                self.token_to_sign = data
            
            # Build reverse map
            self.sign_to_token = {v: k for k, v in self.token_to_sign.items()}
            self.allowed_tokens_list = list(self.token_to_sign.keys())
            print(f"[Vocab] Loaded {len(self.allowed_tokens_list)} tokens")
        else:
            print(f"Warning: Vocab file not found at {VOCAB_PATH} or GCS")

        # Load Aliases
        if os.path.exists(ALIASES_PATH):
            with open(ALIASES_PATH, 'r', encoding='utf-8') as f:
                self.aliases = json.load(f)

    def canon(self, text: str) -> str:
        """Canonicalize text to uppercase token format."""
        if not text:
            return ""
        # Simple normalization: uppercase and strip. 
        # Could handle more complex punctuation stripping if needed.
        return text.strip().upper()

    def apply_aliases(self, token: str) -> str:
        """Resolve aliases like 'PLS' -> 'PLEASE'."""
        token = self.canon(token)
        return self.aliases.get(token, token)

    def token_to_video_name(self, token: str) -> Optional[str]:
        """Get the folder name for a given gloss token."""
        # Check alias first? Valid tokens might be aliases too?
        # Usually we expect inputs to be valid tokens, but let's be safe
        token = self.apply_aliases(token)
        return self.token_to_sign.get(token)

    def video_name_to_token(self, sign_name: str) -> Optional[str]:
        """Get gloss token for a folder name."""
        return self.sign_to_token.get(sign_name)

    def get_allowed_tokens(self, text_context: str = "") -> List[str]:
        """
        Return list of allowed tokens for Gemini.
        Currently returns ALL tokens as the list is manageable (~1800).
        Future optimization: filter based on text_context keywords.
        """
        return self.allowed_tokens_list

    def validate_token(self, token: str) -> bool:
        """Check if a token exists in the vocabulary."""
        token = self.apply_aliases(token)
        return token in self.token_to_sign

# Global instance for easy import
vocab = VocabLoader()
