import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.planner import build_render_plan
from backend.vocab import vocab

def run_test():
    print("Testing Planner Module...")
    
    # 1. Define a dummy gloss
    # Use tokens we know exist from vocab.json (SUPPER, PLEASE)
    gloss = ["SUPPER", "PLEASE", "UNKNOWN_TOKEN_123"]
    
    print(f"Input Gloss: {gloss}")
    
    # 2. Build Plan
    plan = build_render_plan(gloss)
    
    # 3. Print Plan
    import json
    print(json.dumps(plan, indent=2))
    
    # 4. Assertions
    assert len(plan) == 3
    
    # Item 0: SUPPER
    assert plan[0]["token"] == "SUPPER"
    assert plan[0]["type"] == "sign"
    assert "supper" in plan[0]["sign_name"]
    assert plan[0]["assets"]["gif"].endswith("supper/supper.gif")
    
    # Item 1: PLEASE (Alias check? PLS -> PLEASE -> please)
    # If we pass "PLS", planner relies on caller (gemini) to output valid GLOSS.
    # But gemini might output "PLEASE". 
    # Vocab.token_to_video_name handles aliases, so "PLS" would also work if passed.
    assert plan[1]["token"] == "PLEASE"
    assert plan[1]["type"] == "sign"
    
    # Item 2: UNKNOWN
    assert plan[2]["token"] == "UNKNOWN_TOKEN_123"
    assert plan[2]["type"] == "text"
    assert plan[2]["sign_name"] is None
    
    print("\nPlanner Tests Passed!")

if __name__ == "__main__":
    run_test()
