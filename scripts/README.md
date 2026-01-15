# Scripts

Utility scripts for testing and verifying the Unmute backend API.

## verify_api.py

Tests the backend API endpoints to ensure they're working correctly.

### Usage

**Basic usage (uses default localhost:8000):**
```bash
python scripts/verify_api.py
```

**Using custom API URL:**
```bash
# Via environment variable
API_BASE_URL=http://localhost:8001 python scripts/verify_api.py

# For Docker container
API_BASE_URL=http://unmute-backend:8000 python scripts/verify_api.py
```

**Test only health endpoint:**
```bash
python scripts/verify_api.py health
```

### Environment Variables

- `API_BASE_URL` - Base URL of the API to test (default: `http://127.0.0.1:8000`)

### Examples

**Test local development server:**
```bash
python scripts/verify_api.py
```

**Test Dockerized backend:**
```bash
API_BASE_URL=http://localhost:8000 python scripts/verify_api.py
```

**Test remote deployment:**
```bash
API_BASE_URL=https://api.example.com python scripts/verify_api.py
```

**Using .env file:**
```bash
# Create .env file
echo "API_BASE_URL=http://localhost:8000" > .env

# Load and run (bash/zsh)
export $(cat .env | xargs) && python scripts/verify_api.py
```

### Output

The script will:
1. Check the `/health` endpoint
2. Test the `/api/translate` endpoint with sample text
3. Display JSON responses
4. Exit with code 0 on success, 1 on failure

Example output:
```
Using API Base URL: http://127.0.0.1:8000
(Override with API_BASE_URL environment variable)

Testing http://127.0.0.1:8000/health...
{'status': 'ok', 'vocab_size': 717}
✅ Health Check Passed
Testing http://127.0.0.1:8000/api/translate with text='I want apple'...
{
  "gloss": ["I", "WANT", "APPLE"],
  "unmatched": [],
  "plan": [...]
}
✅ Translate Check Passed
```
