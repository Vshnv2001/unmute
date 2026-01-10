import os
import glob
import json
import argparse
import re

def load_all_metadata(dataset_root):
    """Yields (sign_name, meta_dict) for all signs."""
    sign_dirs = glob.glob(os.path.join(dataset_root, "*"))
    for sd in sign_dirs:
        if not os.path.isdir(sd):
            continue
        sign_name = os.path.basename(sd)
        json_path = os.path.join(sd, f"{sign_name}.json")
        # Fallback if json not named after dir? (Task says assume 1 json)
        # We'll try specific name first, then any json
        if not os.path.exists(json_path):
             jsons = glob.glob(os.path.join(sd, "*.json"))
             if jsons:
                 json_path = jsons[0]
             else:
                 json_path = None
        
        meta = {}
        if json_path:
            try:
                with open(json_path, 'r') as f:
                    meta = json.load(f)
            except:
                pass
        
        yield sign_name, meta

def canon_token(text):
    """
    Normalize text to token format.
    - Upper case
    - Replace spaces/hyphens with UNDERSCORE
    - Remove other punctuation
    """
    if not text:
        return ""
    # To upper
    t = text.upper()
    # Replace separators with _
    t = re.sub(r"[\s\-]+", "_", t)
    # Remove non-alphanumeric (except underscore)
    t = re.sub(r"[^A-Z0-9_]", "", t)
    return t

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="sgsl_dataset", help="Path to input dataset")
    parser.add_argument("--output", default="sgsl_processed", help="Path to output directory")
    args = parser.parse_args()
    
    os.makedirs(args.output, exist_ok=True)
    
    token_to_sign = {}
    sign_to_token = {}
    
    print("Building vocabulary...")
    count = 0
    
    for sign_folder, meta in load_all_metadata(args.dataset):
        # Primary source: folder name? or meta 'sign' field?
        # Folder name is unique ID. 'sign' field is display name.
        # We want tokens to match speech.
        
        # Strategy: 
        # 1. Use meta['sign'] as primary token source if available.
        # 2. Use folder name as fallback.
        # 3. Handle duplicates?
        
        raw_name = meta.get('sign', sign_folder)
        token = canon_token(raw_name)
        
        if not token:
            continue
            
        # Collision check?
        if token in token_to_sign:
            # If collision, we might overwrite or ignore. 
            # For MVP, let's keep the one that matches folder name if possible?
            # Or just warn.
            pass
            
        token_to_sign[token] = sign_folder
        sign_to_token[sign_folder] = token
        count += 1
        
    # Validation
    print(f"Mapped {count} signs to tokens.")
    print(f"Unique tokens: {len(token_to_sign)}")
    
    vocab_out = {
        "token_to_sign": token_to_sign,
        "sign_to_token": sign_to_token
    }
    
    out_path = os.path.join(args.output, "vocab.json")
    with open(out_path, 'w') as f:
        json.dump(vocab_out, f, indent=2)
        
    print(f"Saved vocabulary to {out_path}")

if __name__ == "__main__":
    main()
