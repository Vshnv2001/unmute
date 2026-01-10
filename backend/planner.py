from typing import List, Dict, Optional, Any
import os
import sys

# Ensure backend can be imported
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from backend.vocab import vocab

def build_render_plan(gloss_tokens: List[str]) -> List[Dict[str, Any]]:
    """
    Convert a list of gloss tokens into a rendering plan with asset URLs.
    """
    plan = []
    
    for token in gloss_tokens:
        # Default fallback
        item = {
            "token": token,
            "sign_name": None,
            "type": "text", # fallback
            "assets": {}
        }
        
        # Resolve sign name
        sign_name = vocab.token_to_video_name(token)
        
        if sign_name:
            item["sign_name"] = sign_name
            item["type"] = "sign"
            # Construct paths
            # Assuming the web server mounts the root or specific folders to /static
            # Adjust these prefixes based on your actual Flask/FastAPI static mount configuration
            
            # GIF: sgsl_dataset/{sign_name}/{sign_name}.gif
            item["assets"]["gif"] = f"/static/sgsl_dataset/{sign_name}/{sign_name}.gif"
            
            # PKL: sgsl_processed/landmarks_pkl/{sign_name}.pkl
            item["assets"]["pkl"] = f"/static/sgsl_processed/landmarks_pkl/{sign_name}.pkl"
            
        plan.append(item)
        
    return plan
