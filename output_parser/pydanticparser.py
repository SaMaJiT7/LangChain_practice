import os
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel , Field
from typing import Literal, Optional

load_dotenv()

LLM = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-120b",
    task="text-generation",
    do_sample=False,
    repetition_penalty=1.03,
    temperature=0.1, # Keep low for structured data
    max_new_tokens=1024
) # type: ignore

model = ChatHuggingFace(llm=LLM)

class Person(BaseModel):
    name : str = Field(description="Name of the Person.")
    age: int = Field(gt= 18, description="Age of an person in Integer")
    gender: Optional[Literal["M","F","Others"]] = Field(default=None,description="Gender of the Person")
    city : str = Field(description="City of the person from where they belongs to")

parser = PydanticOutputParser(pydantic_object=Person)


template = PromptTemplate(
    template='Generate the name, age, gender and city of a frictional character residing in {place} who is not so Rich.\n {format_instructions}',
    input_variables=['place'],
    partial_variables={'format_instructions' : parser.get_format_instructions()}
) # type: ignore

# prompts = template.invoke({
#     'place':'india'
# })

chains = template | model | parser # type: ignore

model_response = chains.invoke({'place' : 'india'})

print(model_response)