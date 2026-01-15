import sys
import os

# Add project root to path so we can import backend
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.vocab import vocab
def run_tests():
    print("Running Vocabulary Tests...")
    
    # 1. Test Load
    total_tokens = len(vocab.get_allowed_tokens())
    print(f"Total tokens loaded: {total_tokens}")
    assert total_tokens > 0, "Vocab should not be empty"

    # 2. Test Token Lookup (Standard)
    # Using 'SUPPER' based on previous file view of vocab.json
    token = "SUPPER"
    sign_name = vocab.token_to_video_name(token)
    print(f"Token '{token}' -> Sign '{sign_name}'")
    assert sign_name == "supper", f"Expected 'supper', got {sign_name}"

    # 3. Test Inverse Lookup
    rev_token = vocab.video_name_to_token("supper")
    print(f"Sign 'supper' -> Token '{rev_token}'")
    assert rev_token == "SUPPER", f"Expected 'SUPPER', got {rev_token}"

    # 4. Test Aliases
    # "PLS" -> "PLEASE" (Assuming PLEASE is in vocab, it usually is)
    if "PLEASE" in vocab.get_allowed_tokens():
        alias_res = vocab.token_to_video_name("PLS") 
        # We need to know what PLEASE maps to. Let's check.
        real_sign = vocab.token_to_video_name("PLEASE")
        print(f"Alias 'PLS' -> Token 'PLEASE' -> Sign '{alias_res}' (Real: {real_sign})")
        assert alias_res == real_sign, "Alias resolution failed"
    else:
        print("Skipping Alias test 'PLS' -> 'PLEASE' as 'PLEASE' not in vocab")
        
    # 5. Test Non-existent
    assert vocab.token_to_video_name("NON_EXISTENT_TOKEN_XYZ") is None, "Should return None for bad token"

    print("\nAll Tests Passed!")

if __name__ == "__main__":
    run_tests()
