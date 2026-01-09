from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.runnables import RunnableSequence, RunnableParallel
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel,Field
from typing import Optional,Literal

load_dotenv()

# model1 = ChatGoogleGenerativeAI(
#     model="gemini-2.5-flash",
#     max_retries=6,              # Automatically retry up to 6 times
#     request_timeout=60,         # Give it time to wait
# )

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V3.2",
    task="text-generation",  # <--- FIXED: Use hyphen!
    max_new_tokens=250,
    do_sample=False,
    repetition_penalty=1.03,
) # type: ignore

model2 = ChatHuggingFace(llm=llm)

prompt1 = PromptTemplate(
    template = 'Generate a post on LinkedIn about {topic}.',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template = 'Generate a post with sarcasm on Twitter about {topic}.',
    input_variables=['topic']
)


parser = StrOutputParser()


parallel_chain = RunnableParallel({
    'tweet' : RunnableSequence(prompt2, model2, parser),
    'linkedin': RunnableSequence(prompt1, model2, parser)
})

result = parallel_chain.invoke({'topic': 'how you need only to learn 25 dsa to crack any interview.'})

# print(result['linkedin'])
# print("\n")
print(result['tweet'])
