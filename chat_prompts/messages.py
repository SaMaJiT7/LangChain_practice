from langchain.messages import SystemMessage, HumanMessage, AIMessage
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import os

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V3.2",
    task="text-generation",
    do_sample=False,
    repetition_penalty=1.03
) # type: ignore

model = ChatHuggingFace(llm=llm)

messages = [
    SystemMessage(content="You are a helpful research paper summarization assistant."),
    HumanMessage(content="Summarize what is langchain and how it is useful."),
]

response = model.invoke(messages)

messages.append(AIMessage(content=response.content))

print("Chat Messages: ", messages)

