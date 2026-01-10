import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.gemini_client import GeminiClient

def run_test(text):
    print(f"Testing Gemini Client with text: '{text}'")
    client = GeminiClient()
    
    # Check if we have an API key or falling back to mock
    if client.model:
        print("Using Real Gemini API")
    else:
        print("Using Mock Fallback")
        
    result = client.text_to_gloss(text)
    
    print("\n--- Result ---")
    print(result)
    print("--------------\n")
    
    # Basic assertions
    assert "gloss" in result
    assert isinstance(result["gloss"], list)
    assert "unmatched" in result
    
    # Specific checks if using Mock (since reliable)
    if not client.model:
        # Mock just uppercases and matches.
        # "I want apple" -> I (if in vocab), WANT (if in), APPLE (if in)
        # Note: "I" might not be in vocab, "WANT" is "WANT", "APPLE" is "APPLE"
        pass

if __name__ == "__main__":
    text = sys.argv[1] if len(sys.argv) > 1 else "I want to eat apple"
    run_test(text)
