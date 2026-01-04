from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline # type: ignore
import os
from dotenv import load_dotenv

load_dotenv()

# Debug: Check if token is actually loaded
if not os.getenv("HUGGINGFACEHUB_API_TOKEN"):
    print("CRITICAL ERROR: HUGGINGFACEHUB_API_TOKEN not found.")

model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')  # type: ignore


documents = ["how to bake a cake?", "everyone makes gingerbread cookies during christmas", "Making corrsaints is hard", "Do you like applepie or crumblePie?"]

embeddings = model.embed_documents(documents)

print(f"Vector length: {len(embeddings)}")
print("Vector:", str(embeddings))

