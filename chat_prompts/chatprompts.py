from langchain_core.prompts import ChatPromptTemplate
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


chat_template = ChatPromptTemplate([
    ('system', "You are a helpful {Domain} life Guide expert."),
    ('human', "Explain in simple terms about {Topic} and its consequences.")
])

prompts = chat_template.invoke({'Domain': 'Polyamorous Relationships', 'Topic': 'Hotwife lifestyle'})

print("Chat Prompts: ", prompts)