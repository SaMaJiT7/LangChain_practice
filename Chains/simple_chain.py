from langchain_huggingface import HuggingFaceEndpoint , ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


import os

load_dotenv()

LLM = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-120b",
    task="text-generation",
    do_sample=False,
    repetition_penalty=1.03,
    temperature=0.1, # Keep low for structured data
    max_new_tokens=1024
) # type: ignore


prompt = PromptTemplate(
    template='YOu are very knowledgeable and Helpful Assistant.Generate the Five Greatest niche facts about {Topic}',
    input_variables=['Topic']
) # type: ignore

model = ChatHuggingFace(llm=LLM)

parser = StrOutputParser()

chain = prompt | model | parser

model_response = chain.invoke({'Topic' : 'Dublin'})

print(model_response)

chain.get_graph().print_ascii()


