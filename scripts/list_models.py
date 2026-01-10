import os
import google.generativeai as genai
from dotenv import load_dotenv

# Load .env file
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'backend', '.env'))

api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    print("No API Key found")
    exit(1)

genai.configure(api_key=api_key)

print("Listing models...")
for m in genai.list_models():
    if 'generateContent' in m.supported_generation_methods:
        print(m.name)
