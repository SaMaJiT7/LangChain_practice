from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.runnables import RunnableSequence, RunnableParallel,RunnablePassthrough,RunnableLambda,RunnableBranch
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import TextLoader,PyPDFLoader,CSVLoader


load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    max_retries=6,             # Automatically retry up to 6 times
    request_timeout=60,        # Give it time to wait
)


loader = PyPDFLoader('resume.pdf')

docs = loader.lazy_load()

# print(len(docs))
# print(docs[0].page_content)
# print(docs[0].metadata)

for d in docs:
    print(d.metadata)
