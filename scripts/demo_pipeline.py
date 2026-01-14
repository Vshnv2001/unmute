import sys
import os
import json

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.gemini_client import GeminiClient
from backend.planner import build_render_plan
from backend.vocab import vocab
def run_demo(text):
    print(f"\n🎥 Text-to-SGSL Demo Pipeline 🎥")
    print(f"Input Text: \"{text}\"")
    
    # 1. Initialize Client
    client = GeminiClient()
    if not client.model:
        print("⚠️  Warning: Using Mock Gemini Client (Set GEMINI_API_KEY in backend/.env for real AI)")
    
    # 2. Get Gloss from Gemini
    print("\n--- Step 1: Generating Gloss ---")
    gloss_result = client.text_to_gloss(text)
    gloss_tokens = gloss_result.get("gloss", [])
    unmatched = gloss_result.get("unmatched", [])
    
    print(f"Gloss: {gloss_tokens}")
    if unmatched:
        print(f"Unmatched: {unmatched}")
    
    # 3. Build Render Plan
    print("\n--- Step 2: Building Render Plan ---")
    plan = build_render_plan(gloss_tokens)
    
    # 4. Display Result
    print(json.dumps(plan, indent=2))
    
    print("\n✅ Demo Complete!")

if __name__ == "__main__":
    text = sys.argv[1] if len(sys.argv) > 1 else "I want to eat apple"
    run_demo(text)
