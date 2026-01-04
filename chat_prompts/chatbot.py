from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import os

load_dotenv()

if not os.getenv("HUGGINGFACEHUB_API_TOKEN"):
    print("CRITICAL ERROR: HUGGINGFACEHUB_API_TOKEN not found.")

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V3.2",
    task="text-generation",  # <--- FIXED: Use hyphen!
    do_sample=False,
    repetition_penalty=1.03,
) # type: ignore

model = ChatHuggingFace(llm=llm)

chat_history = []

while True:
    user_input = input("You: ")
    chat_history.append(user_input)
    if user_input.lower() in ["exit", "quit"]:
        print("Exiting chat. Goodbye!")
        break
    response = model.invoke(chat_history)
    chat_history.append(response.content)
    print(f"AI: {response.content}")

print("Chat History: ",chat_history)
