from langchain.messages import SystemMessage, HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os
from typing import TypedDict,Annotated,Optional,Literal
from pydantic import BaseModel,Field


load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", # or "gemini-1.5-pro"
    temperature=0.2
)


# schema
json_schema = {
  "title": "Review",
  "type": "object",
  "properties": {
    "key_themes": {
      "type": "array",
      "items": {
        "type": "string"
      },
      "description": "Write down all the key themes discussed in the review in a list"
    },
    "summary": {
      "type": "string",
      "description": "A brief summary of the review"
    },
    "visual_prompt": {
      "type": "string",
      "description": "A creative prompt to generate a thumbnail image for this review"
    },
    "sentiment": {
      "type": "string",
      "enum": ["pos", "neg"],
      "description": "Return sentiment of the review either negative, positive or neutral"
    },
    "pros": {
      "type": ["array", "null"],
      "items": {
        "type": "string"
      },
      "description": "Write down all the pros inside a list"
    },
    "cons": {
      "type": ["array", "null"],
      "items": {
        "type": "string"
      },
      "description": "Write down all the cons inside a list"
    },
    "name": {
      "type": ["string", "null"],
      "description": "Write the name of the reviewer"
    }
  },
  "required": ["key_themes", "summary", "sentiment"]
}


structured_model = model.with_structured_output(json_schema) # type: ignore

review_text = """The new MacBook Air is a stunning piece of jewelry that occasionally functions as a computer. Its biggest "pro" is the breathtakingly thin design, which is so delicate that I’m terrified to put it in a backpack without a military-grade case—a thrill you just don't get with sturdier, cheaper laptops! I also applaud Apple's brave decision to include only two ports; while some might call this a "con," I see it as an opportunity to embrace the "dongle lifestyle," carrying around a messy $80 hub everywhere I go just to plug in a mouse. It really makes you appreciate the days when computers were practical.

Performance-wise, the fanless design is a masterpiece of silence. Instead of annoying you with fan noise, the laptop thoughtfully manages heat by simply slowing down to a crawl when you try to export a video, effectively forcing you to take a break while the chassis warms your lap like a radiator. And let’s talk about the base 8GB of RAM: critics call it "obsolete," but I call it a "focus feature." By freezing the system whenever I open more than six tabs, the MacBook actively prevents me from multitasking, ensuring I stay disciplined. It’s the most expensive, beautiful web browser money can buy!
                                         
Review by Samajit Nandi
"""

print("1. Analyzing Text & Designing Image...")
model_response = structured_model.invoke(review_text)


#print(f"\n[Generated Prompt]: {model_response.visual_prompt}") # type: ignore
print("visual_prompt: ",model_response['visual_prompt']) # type: ignore



