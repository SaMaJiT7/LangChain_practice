from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
import requests
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)

@tool
def addition(a: int, b: int) -> int:
    """Adds two numbers together."""
    return a + b


llm_with_tool = model.bind_tools([addition])
query = HumanMessage('what is the sum of 49 and 99')
messages = [query]
result = llm_with_tool.invoke(messages)
messages.append(result) # type: ignore
input = result.tool_calls[0]
final_result = addition.invoke(input)
messages.append(final_result)
# print(final_result)
# print(messages)
print(llm_with_tool.invoke(messages).content)