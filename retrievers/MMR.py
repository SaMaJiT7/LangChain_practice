from pinecone import Pinecone, ServerlessSpec
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
import os
import sys
from dotenv import load_dotenv

# Sample documents regarding modern dating culture
docs = [
    Document(page_content="Modern dating often relies on algorithmic matching via apps like Hinge and Bumble."),
    Document(page_content="Ghosting is the act of suddenly ending communication without any explanation."),
    Document(page_content="A situationship is an undefined romantic arrangement that lacks clear commitment."),
    Document(page_content="Red flags are warning signs indicating unhealthy behavior in a potential partner."),
    Document(page_content="Love bombing involves overwhelming someone with excessive affection and attention early on."),
    Document(page_content="Breadcrumbing is sending sporadic messages to keep someone interested without intending to pursue them."),
]

embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vectorstore = FAISS.from_documents(docs, embedding=embedding)

retreiver = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 3, "lambda_mult": 0.5})

query = "what is ghosting in dating?"

result = retreiver.invoke(query)


for i, doc in enumerate(result):
    print(f"Document {i+1}:\n")
    print(doc.page_content)
    print("\n"+"-"*80+"\n")




