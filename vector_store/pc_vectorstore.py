from pinecone import Pinecone, ServerlessSpec
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document
from history_docs import HISTORY_DOCUMENTS
import os
from dotenv import load_dotenv

load_dotenv()

embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

pinecone_key = os.getenv("PINECONE_API_KEY")

docs = HISTORY_DOCUMENTS

pc = Pinecone()
index_name = "langchain-test-index"

existing_indexes = [index.name for index in pc.list_indexes()]
if index_name not in existing_indexes:
    pc.create_index(
        name=index_name,
        dimension=384, # Must match the embedding model (MiniLM=384, OpenAI=1536)
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
# uploading documents to Pinecone
# print("Uploading documents to Pinecone Index...")
# vector_store = PineconeVectorStore.from_documents(
#     documents=docs,
#     embedding=embeddings,
#     index_name=index_name
# )
# print("Upload Complete.")

#  Retrieving documents from Pinecone

vector_store = PineconeVectorStore(
    embedding=embeddings,
    index_name=index_name
)
doc6 = Document(
    page_content= "Nazi Germany refers to the period from 1933 to 1945 when Adolf Hitler and the National Socialist German Workers' Party governed Germany. The regime established a totalitarian state, suppressed political opposition, controlled the media, and promoted extreme nationalist and racist ideology. Nazi policies led to World War II and the Holocaust, in which approximately six million Jews and millions of other civilians were systematically murdered.",
    metadata= {
    "topic": "world history",
    "regime": "Nazi Germany",
    "time_period": "1933–1945",
    "region": "Germany",
    "key_figures": ["Adolf Hitler"],
    "historical_significance": [
      "World War II",
      "Holocaust",
      "totalitarian rule"
    ],
    "content_type": "educational"
}
)
vector_store.add_documents([doc6])
# print("Connected to Pinecone Database.")

# retrieve
query = "who can be responsible for world war II and why?"

# results = vector_store.similarity_search(query=query, k=3)

# results = vector_store.similarity_search_with_score(query=query, k=1)
updated_document = Document(
    page_content= "The Indian Independence Movement was a prolonged struggle against British colonial rule, culminating in independence in 1947. Leaders like Mahatma Gandhi promoted nonviolent resistance, which became a powerful method of political protest worldwide. The movement also saw significant contributions from other leaders such as Jawaharlal Nehru, Subhas Chandra Bose, and Sardar Vallabhbhai Patel, who played crucial roles in mobilizing the masses and negotiating with the British government.",
    metadata={
        "topic": "indian history",
        "event": "Indian Independence",
        "year": 1947,
        "key_figures": ["Netaji subhash chandra bose","Mahatma Gandhi","Jawaharlal Nehru","Sardar Vallabhbhai Patel"],
        "method": "nonviolent resistance"
    }
)
# Update document by deleting and re-adding
vector_store.delete(ids=["fd621fb5-c5bb-4c80-85f1-c985da273b29"])
vector_store.add_documents([updated_document], ids=["fd621fb5-c5bb-4c80-85f1-c985da273b29"])

results = vector_store.similarity_search_with_score(query="", filter={"event": "Indian Independence"},k=1)
print(results)

# type: ignore
# print("Top 3 results:")
# for i, doc in enumerate(results, 1):
#     print(f"\n--- Result {i} ---")
#     print(f"Content: {doc.page_content[:200]}...")
#     print(f"Metadata: {doc.metadata}")