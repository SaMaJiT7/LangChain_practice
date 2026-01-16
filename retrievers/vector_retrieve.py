from pinecone import Pinecone, ServerlessSpec
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document
import os
import sys
from dotenv import load_dotenv

# Add parent directory to path to allow imports from sibling folders
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vector_store.history_docs import HISTORY_DOCUMENTS

load_dotenv()

os.environ["PINECONE_API_KEY"]

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


pc = Pinecone()
index_name = "langchain-test-index"

vector_store = PineconeVectorStore.from_documents(
    embedding=embedding_model,
    index_name=index_name,
    documents=HISTORY_DOCUMENTS
)

retriever = vector_store.as_retriever(search_kwargs={"k": 2})


query = "The word utopian was introduced in france in which century?"
results = retriever.invoke(query)

for i, doc in enumerate(results):
    print(f"Document {i+1}:\n")
    print(doc.page_content)
    print("\n"+"-"*80+"\n")
