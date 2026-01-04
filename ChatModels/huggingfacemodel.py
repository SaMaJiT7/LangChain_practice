from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint # type: ignore
import os
from dotenv import load_dotenv

# 1. Load Environment
load_dotenv()

# Debug: Check if token is actually loaded
if not os.getenv("HUGGINGFACEHUB_API_TOKEN"):
    print("CRITICAL ERROR: HUGGINGFACEHUB_API_TOKEN not found.")

# 2. Create the Endpoint (The "Engine")
llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V3.2",
    task="text-generation",  # <--- FIXED: Use hyphen!
    max_new_tokens=250,
    do_sample=False,
    repetition_penalty=1.03,
) # type: ignore

# 3. Create the Chat Wrapper (The "Translator")
model = ChatHuggingFace(llm=llm)

# 4. Run
try:
    print("Sending request to Hugging Face...")
    model_response = model.invoke("how should i approach my crush for a confession?")
    print("\nAnswer:")
    print(model_response.content)
except Exception as e:
    print(f"\nError: {e}")
    print("Tip: If you got a 503 or StopIteration, try changing repo_id to 'HuggingFaceH4/zephyr-7b-beta'")