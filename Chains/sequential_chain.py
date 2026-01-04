from langchain_huggingface import HuggingFaceEndpoint , ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

LLM = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-120b",
    task="text-generation",
    do_sample=False,
    repetition_penalty=1.03,
    temperature=0.1, # Keep low for structured data
    max_new_tokens=1024
) # type: ignore


prompt1 = PromptTemplate(
    template='YOu are very knowledgeable and Helpful Assistant.' \
    'Generate a report on the fall of {place} during or before Worldwar II',
    input_variables=['place']
) # type: ignore


prompt2 = PromptTemplate(
    template= 'Generate the most important and good five facts from the following Report ' \
    '{text}',
    input_variables=['text']
) # type: ignore

model = ChatHuggingFace(llm=LLM)

parser = StrOutputParser()

chain = prompt1 | model | parser | prompt2 | model | parser

model_response = chain.invoke({'place' : 'Nazi Germany'})

print(model_response)


