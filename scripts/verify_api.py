import requests
import json
import sys

BASE_URL = "http://127.0.0.1:8000"

def test_health():
    print(f"Testing {BASE_URL}/health...")
    try:
        res = requests.get(f"{BASE_URL}/health")
        print(res.json())
        assert res.status_code == 200
        print("✅ Health Check Passed")
    except Exception as e:
        print(f"❌ Health Check Failed: {e}")
        sys.exit(1)

def test_translate(text="I want apple"):
    print(f"Testing {BASE_URL}/api/translate with text='{text}'...")
    try:
        res = requests.post(f"{BASE_URL}/api/translate", json={"text": text})
        data = res.json()
        print(json.dumps(data, indent=2))
        assert res.status_code == 200
        assert "gloss" in data
        assert "plan" in data
        print("✅ Translate Check Passed")
    except Exception as e:
        print(f"❌ Translate Check Failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "health":
        test_health()
    else:
        test_health()
        test_translate()
