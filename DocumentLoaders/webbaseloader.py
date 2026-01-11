from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.runnables import RunnableSequence, RunnableParallel,RunnablePassthrough,RunnableLambda,RunnableBranch
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import TextLoader,PyPDFLoader,WebBaseLoader

load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    max_retries=6,             # Automatically retry up to 6 times
    request_timeout=60,        # Give it time to wait
)

loader = WebBaseLoader("https://www.amazon.in/Apple-MacBook-13-inch-10-core-Unified/dp/B0DZDDKTQZ/ref=pd_ci_mcx_mh_mcx_views_0_image")

docs = loader.load()

prompt = PromptTemplate(
    template="Answer the question based on the context below:\n\nContext: {context}\n\nQuestion: {question}\n\nAnswer:",
    input_variables=["context", "question"], # type: ignore
)

parser = StrOutputParser()

print(len(docs))

chains = prompt | model | parser

result = chains.invoke({
    "context": docs[0].page_content,
    "question": "What is the one disadvantage of this laptop mentioned in the description?"
})

print(result)
