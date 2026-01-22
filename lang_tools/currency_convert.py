from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from langchain_core.tools import InjectedToolArg
from typing import Annotated
import requests
from dotenv import load_dotenv
import os
import json

load_dotenv()

API_KEY = os.getenv("EXCHANGE_RATE_API_KEY")

chat_model = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

@tool
def convert_currency(Basecurrency: str, Targetcurrency: str) -> float:
    """Fetches the current exchange rate between two currencies. Use this to get the conversion rate from one currency to another (e.g., USD to INR)."""

    currency_fixes = {"EURO": "EUR", "DOLLAR": "USD", "RUPEE": "INR", "POUND": "GBP"}
    Basecurrency = currency_fixes.get(Basecurrency.upper(), Basecurrency.upper())
    Targetcurrency = currency_fixes.get(Targetcurrency.upper(), Targetcurrency.upper())

    url = f"https://v6.exchangerate-api.com/v6/{API_KEY}/pair/{Basecurrency}/{Targetcurrency}"
    response = requests.get(url)
    return response.json()


@tool
def convert(base_currency: float, conversion_rate: Annotated[float, InjectedToolArg]) -> float:
    """Calculates the converted amount by multiplying the base currency amount with the conversion rate."""
    converted_amount = base_currency * conversion_rate
    return converted_amount


llm_with_tool = chat_model.bind_tools([convert_currency, convert])

messages = [HumanMessage(' what is the conversion rate from pound to INR and convert 150 pound to INR?')]

ai_message = llm_with_tool.invoke(messages)

print(f"AI Message: {ai_message}")
print(f"AI Content: {ai_message.content}")

messages.append(ai_message) # type: ignore

Tools = ai_message.tool_calls
print(f"Tool calls: {Tools}")

if not Tools:
    print("No tool calls were made by the LLM")

conversion_rate = None

for tool_call in Tools:
    if tool_call["name"] == "convert_currency":
        tool_message_1 = convert_currency.invoke(tool_call)
        response_data = json.loads(tool_message_1.content)
        if "conversion_rate" in response_data:
            conversion_rate = response_data['conversion_rate']
        else:
            print(f"API Error: {response_data}")
        messages.append(tool_message_1)

for tool_call in Tools:
    if tool_call["name"] == "convert":
        if conversion_rate is None:
            raise ValueError("Conversion rate not available. convert_currency must be called first.")
        tool_call["args"]["base_currency"] = float(tool_call["args"]["base_currency"])
        tool_call["args"]["conversion_rate"] = conversion_rate
        tool_message2 = convert.invoke(tool_call)
        messages.append(tool_message2)

llm_response = llm_with_tool.invoke(messages)
print(llm_response.content)