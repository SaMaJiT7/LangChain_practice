from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

LLM = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-120b",
    task="text-generation",
    do_sample=False,
    repetition_penalty=1.03,
    temperature=0.7
) # type: ignore


model = ChatHuggingFace(llm=LLM)


#Prompt one for detail one
template1 = PromptTemplate(
    template="Write a Detailed Report the {topic}, if possible includes external references also and make it little for beginer to understand.",
    input_variables=['topic']
) # type: ignore

#Prompt two for summary of the prompt one
template2 = PromptTemplate(
    template="Write a Simple but effective five lines summary on the following text. /n {txt}",
    input_variables=['txt']
) # type: ignore


parser = StrOutputParser()

chain = template1 | model | parser | template2 | model | parser

response = chain.invoke({'topic' : 'Fyodor Dostoevsky'})

print(response)



