from langchain.messages import SystemMessage, HumanMessage, AIMessage
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import os
from typing import TypedDict,Annotated,Optional,Literal


load_dotenv()

LLM = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-120b",
    task="text-generation",
    do_sample=False,
    repetition_penalty=1.03
) # type: ignore

model = ChatHuggingFace(llm=LLM)

#schema
class Review(TypedDict):
    Key_themes: Annotated[list[str], "Include all the necessary and relevant themes discussed in the review"]
    summary: Annotated[str, "A Brief summary of the review in shortest possible way"]
    sentiment: Annotated[Literal["Positive","Negative"], "The overall sentiment of the review , can be positive or negative"]
    pros: Annotated[Optional[list[str]], "List all reviews pros what is mentioned in the review"]
    cons: Annotated[Optional[list[str]], "List all revies cons what is mentioned in the review"]
    name: Annotated[Optional[str], "write the full name of the reviewer"]



structured_model = model.with_structured_output(Review) # type: ignore

model_response = structured_model.invoke("""The new MacBook Air is a stunning piece of jewelry that occasionally functions as a computer. Its biggest "pro" is the breathtakingly thin design, which is so delicate that I’m terrified to put it in a backpack without a military-grade case—a thrill you just don't get with sturdier, cheaper laptops! I also applaud Apple's brave decision to include only two ports; while some might call this a "con," I see it as an opportunity to embrace the "dongle lifestyle," carrying around a messy $80 hub everywhere I go just to plug in a mouse. It really makes you appreciate the days when computers were practical.

Performance-wise, the fanless design is a masterpiece of silence. Instead of annoying you with fan noise, the laptop thoughtfully manages heat by simply slowing down to a crawl when you try to export a video, effectively forcing you to take a break while the chassis warms your lap like a radiator. And let’s talk about the base 8GB of RAM: critics call it "obsolete," but I call it a "focus feature." By freezing the system whenever I open more than six tabs, the MacBook actively prevents me from multitasking, ensuring I stay disciplined. It’s the most expensive, beautiful web browser money can buy!
                                         
Review by Samajit Nandi
""")

print(model_response["sentiment"]) # type: ignore
print(model_response["name"]) # type: ignore
