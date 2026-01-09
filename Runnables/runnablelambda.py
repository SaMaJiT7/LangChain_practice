from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.runnables import RunnableSequence, RunnableParallel,RunnablePassthrough,RunnableLambda
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


def clean_text(text: str) -> str:
    text = text.replace("\n", " ")
    text = text.replace("\'s"," ")
    text = text.strip()
    return len(text) # type: ignore

parser = StrOutputParser()

joke_generation = RunnableSequence(prompt, model, parser)

Parallel_chain = RunnableParallel({
    'joke': RunnablePassthrough(),
    'word_count': RunnableLambda(lambda x: clean_text(x))
})

final_chain = RunnableSequence(joke_generation, Parallel_chain)

print(final_chain.invoke({'topic': 'how easy is to get laid in bangalore women'}))
