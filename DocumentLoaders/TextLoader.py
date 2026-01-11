from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.runnables import RunnableSequence, RunnableParallel,RunnablePassthrough,RunnableLambda,RunnableBranch
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel,Field
from typing import Optional,Literal
from langchain_community.document_loaders import TextLoader # type: ignore


load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    max_retries=6,             # Automatically retry up to 6 times
    request_timeout=60,        # Give it time to wait
)
loader = TextLoader('life.txt', encoding='utf-8') # type: ignore

docs = loader.load()

prompt = PromptTemplate(
    template="Write a summary for the following poem -\n {poem}",
    input_variables=['poem'] # type: ignore
)

parser = StrOutputParser()

summary_chain = prompt | model | parser
result = summary_chain.invoke({'poem': docs[0].page_content})

print(docs[0].page_content)
print(docs[0].metadata)
print(result)