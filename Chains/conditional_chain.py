from langchain_huggingface import HuggingFaceEndpoint , ChatHuggingFace
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableBranch,RunnableParallel,RunnableLambda
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel,Field
from typing import Optional,Literal

load_dotenv()

model = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.8
    )


class Review(BaseModel):
    name : Optional[str] = Field(description="Name of the Person.")
    review: Optional[str] = Field(description="Brief summary of the whole review")
    sentiment: Literal["POSITIVE","NEGATIVE"] = Field(description="Sentiment of the Review like Positive or Negative")
    Product : Optional[str] = Field(description="Product Name or Type on which the Whole Review is written.")


parser = StrOutputParser()

outparser = PydanticOutputParser(pydantic_object=Review)

feedback_prompt = PromptTemplate(
    template='Classify the given feedback received from an customer into Positive or Negative. The Feedback is {feedback}. \n {format_instructions}',
    input_variables=['feedback'],
    partial_variables = {'format_instructions' : outparser.get_format_instructions()} # type: ignore
)

classification_chain = feedback_prompt | model | outparser

# model_response = classification_chain.invoke({'feedback':'This is a okayish SmartPhone to use in this modern time.'})

# print(model_response.sentiment)

positive_prompt = PromptTemplate(
    template='Generate a Humble and thankful response to the {feedback},Appreciate them for their review and also ask for more feedback like rating upon five stars.',
    input_variables=['feedback']
)

negative_prompt = PromptTemplate(
    template='Generate a Humble and merciness response to the {feedback},Asking them  to pardon us for the  issue and also try to ask them to contact via Email.',
    input_variables=['feedback']
)

branch_chain = RunnableBranch(
    (lambda x:x.sentiment == 'POSITIVE', positive_prompt | model | parser),# type: ignore
    (lambda x:x.sentiment == 'NEGATIVE', negative_prompt | model | parser), # type: ignore
    RunnableLambda(lambda x: "could not find sentiment")
)

chains = classification_chain | branch_chain

result = chains.invoke({'feedback':'I’ll be honest, I was initially skeptical about buying from this seller because I saw some negative Google reviews. But my experience turned out to be completely positive! Everything arrived in perfect condition — nothing was damaged, and I didn’t receive anything fake. The MacBook Air M4 I ordered is not 100% original and bad build quality.Coming from a Windows background, I was unsure at first, but after using it for just a couple of hours, I was so disappointed by how less userfriendly and laggy it is. Now I actually dislike using Windows — the difference is that big! The performance, build quality, and overall experience are okayish.If you’re on the fence, don’t worry — just go for it. It’s a must-buy if you want to waste your money and not really totally worth the money!'})

print(result)

chains.get_graph().print_ascii()





