from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.runnables import RunnableSequence, RunnableParallel,RunnablePassthrough,RunnableLambda,RunnableBranch
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel,Field
from typing import Optional,Literal

load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    max_retries=6,              # Automatically retry up to 6 times
    request_timeout=60,
)

prompt = PromptTemplate(
    template="Generate a Report/Review on the {topic}",
    input_variables=['topic']
)

promptnext = PromptTemplate(
    template="Summarize the following report/review in short: {report}",
    input_variables=['report']
)
parser = StrOutputParser()

report_chain = RunnableSequence(prompt, model, parser)

branch_chain = RunnableBranch(
    (lambda x: len(x.split()) > 500, RunnableSequence(promptnext, model, parser)), # type: ignore
    RunnablePassthrough()
)


final_chain = RunnableSequence(report_chain, branch_chain)

print(final_chain.invoke({'topic': 'How easy to find a women in Russia and impress her.'}))