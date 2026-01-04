import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

# 1. Load the environment variables
load_dotenv()

# 2. explicit fetch (Debugging step: Print this to see if it's actually loaded!)
my_key = os.getenv("GEMINI_API_KEY")

if not my_key:
    print("ERROR: API Key not found. Check your .env file.")
else:
    # 3. Initialize with the key explicitly passed
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=my_key,
        temperature=0.8
    )

    # 4. Ask a static question (Gemini can answer this without internet)
    # Changed query to something the model DOES know
    query = "What are the standard precautions for Severe AQI levels (400+)?"
    
    result = llm.invoke(query)

    # 5. Print ONLY the content
    print("Response:\n", result.content)