import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("Error: API Key not found in environment!")
else:
    genai.configure(api_key=api_key) # type: ignore
    
    print("Attempting to list available models...")
    try:
        # List all models available to your key
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                print(f"Found Model: {m.name}")
    except Exception as e:
        print(f"FAILED to list models. Error:\n{e}")