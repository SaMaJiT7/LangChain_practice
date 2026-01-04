from langchain_openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("OpenAI_API")

client = OpenAI(model="gpt-3.5-turbo", api_key=API_KEY) # type: ignore

result = client.invoke("What is the AQI of new york city today?")

print(result)


