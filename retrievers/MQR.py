from pinecone import Pinecone, ServerlessSpec
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_google_genai import ChatGoogleGenerativeAI
import os
import sys
from dotenv import load_dotenv

load_dotenv()

docs = [
    # --- Health & Fitness Documents ---
    Document(
        page_content="Progressive overload acts as the fundamental principle for building muscle and strength.", 
        metadata={"category": "fitness", "id": "H1"}
    ),
    Document(
        page_content="A caloric deficit is scientifically required to lose body fat, regardless of the diet type.", 
        metadata={"category": "fitness", "id": "H2"}
    ),
    Document(
        page_content="Creatine monohydrate is a widely researched supplement that boosts physical performance.", 
        metadata={"category": "fitness", "id": "H3"}
    ),
    Document(
        page_content="Consuming sufficient protein post-workout aids in muscle recovery and hypertrophy.", 
        metadata={"category": "fitness", "id": "H4"}
    ),

    # --- Twitch Streaming Documents ---
    Document(
        page_content="OBS Studio is the most popular open-source software used for broadcasting on Twitch.", 
        metadata={"category": "streaming", "id": "T1"}
    ),
    Document(
        page_content="Twitch chat sentiment can fluctuate rapidly depending on the streamer's gameplay performance.", 
        metadata={"category": "streaming", "id": "T2"}
    ),
    Document(
        page_content="Raiding is a feature that allows a streamer to send their live viewers to another channel.", 
        metadata={"category": "streaming", "id": "T3"}
    ),
    Document(
        page_content="Parasocial relationships often form when viewers feel a close personal bond with a streamer.", 
        metadata={"category": "streaming", "id": "T4"}
    ),
]

embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vectorstore = FAISS.from_documents(documents=docs, embedding=embedding)


simple_retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.1,
)
multiquery_retriever = MultiQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(search_kwargs={"k": 6}),
    llm=llm
)

query = "how to build muscle and maintain a caloric deficit so that i dont feel less energy through the day when i am streaming?"

simple_results = simple_retriever.invoke(query)
multiquery_results = multiquery_retriever.invoke(query)

for i, doc in enumerate(simple_results):
    print(f"\n--- Result {i+1} ---")
    print(doc.page_content)

print("*"*150)

for i, doc in enumerate(multiquery_results):
    print(f"\n--- Result {i+1} ---")
    print(doc.page_content)