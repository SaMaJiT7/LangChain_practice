from langchain_huggingface import HuggingFaceEmbeddings
import os 
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv()

if not os.getenv("HUGGINGFACEHUB_API_TOKEN"):
    print("CRITICAL ERROR: HUGGINGFACEHUB_API_TOKEN not found.")


model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')  # type: ignore

documents = [
    "Twitch streamers build communities by engaging with viewers through live chat, creating a highly interactive entertainment experience.",
    "Many successful streamers diversify their content across platforms like YouTube, Instagram, and TikTok to grow their audiences.",
    "Streamers earn revenue through subscriptions, donations, ad revenue, sponsorships, and affiliate links.",
    "Twitch has various categories beyond gaming, including IRL chats, music, fitness, cooking, and educational streams.",
    "Consistency in streaming schedule is one of the strongest predictors of growth for new streamers.",
    "Twitch streamers can receive special recognition such as Partner status, which grants them additional monetization options and platform perks."
]

documents_vectors = model.embed_documents(documents)

query = "How much and what are the ways Twitch streamers earn money?"

query_vector = model.embed_query(query)

cosine_score = cosine_similarity([query_vector], documents_vectors)[0] # type: ignore

index, score = sorted(list(enumerate(cosine_score)), key=lambda x:x[1])[-1]

print("Cosine Similarity Scores::",cosine_score)

print("\nQuery:\n", query)
print(f"\nMost similar document (Score: {score}):\n{documents[index]}")


