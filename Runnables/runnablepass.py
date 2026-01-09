from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.runnables import RunnableSequence, RunnableParallel,RunnablePassthrough
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
    request_timeout=60,         # Give it time to wait
)

prompt = PromptTemplate(
    template='Generate a funny and adult humour joke about {topic} to impress this older women.',
    input_variables=['topic']
)

explainprompt = PromptTemplate(
    template='Explain the followinng joke you generated: {joke} in short manner, not more than 50 words.',
    input_variables=['joke']# type: ignore
)

parser = StrOutputParser()

joke_gen_chain = RunnableSequence(prompt, model, parser)

parallel_chain = RunnableParallel(
    {
        'joke': RunnablePassthrough(), # type: ignore
        'explanation': RunnableSequence(explainprompt,model,parser) # type: ignore
    }
)

final_chain = RunnableSequence(joke_gen_chain, parallel_chain)


Result = final_chain.invoke({'topic': 'how easy is to get older women into your bed.'})

print(Result)