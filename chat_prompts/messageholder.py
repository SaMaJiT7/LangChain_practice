from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
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
    ('system', "You are a helpful Customer Support AI assistant. You are very polite and attentive to every details of the customer, you always try to make the customer happy and satisfied with the service provided."),
    MessagesPlaceholder(variable_name='Chat_History'),
    ('human', "Customer: {Customer_Query}"),

]) # type: ignore

# 3. THE FIX: Custom Parser for your file format
chat_history = []

try:
    with open('Chat_History.txt', 'r') as f:
        lines = f.readlines()
        
        for line in lines:
            line = line.strip()
            
            # Skip empty lines
            if not line:
                continue

            # Extract content between content=" AND the last quote "
            # This handles the trailing '), or ')
            if 'content="' in line:
                start_index = line.find('content="') + len('content="')
                end_index = line.rfind('"') # Find the LAST quote in the line
                
                # Extract the actual message text
                extracted_text = line[start_index:end_index]
                
                if line.startswith("HumanMessage"):
                    chat_history.append(HumanMessage(content=extracted_text))
                elif line.startswith("AIMessage"):
                    chat_history.append(AIMessage(content=extracted_text))

except FileNotFoundError:
    print("Error: Chat_History.txt not found. Using empty history.")

print(f"Successfully loaded {len(chat_history)} messages from history.")


print("ChatHistory: ", chat_history)

prompt = chat_template.invoke({'Chat_History': chat_history, 'Customer_Query':'I have an issue with my order, it did not arrived yet. Can you help me with a update?, The order number is ##5544.'})

model_response = model.invoke(prompt)
chat_history.append(AIMessage(content=model_response.content))

print("AI Response: ", model_response.content)

