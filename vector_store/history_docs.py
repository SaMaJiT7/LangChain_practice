from pinecone import Pinecone, ServerlessSpec
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISSs one of the world's earliest urban civilizations, flourishing around 2500 BCE in the northwestern regions of South Asia. Major cities like Harappa and Mohenjo-daro were known for advanced urban planning, drainage systems, and standardized brick construction.",
from langchain.retrievers.multi_query import MultiQueryRetriever
import osc": "ancient history",
import sysization": "Indus Valley",
from dotenv import load_dotenv
    "region": "South Asia",
load_dotenv()ty": "beginner"
    }
docs = [
    # --- Health & Fitness Documents ---
    Document(nt(
        page_content="Progressive overload acts as the fundamental principle for building muscle and strength.", 476 CE in the West. It was known for its legal systems, military organization, road networks, and architectural achievements such as aqueducts and amphitheaters.",
        metadata={"category": "fitness", "id": "H1"}
    ),opic": "classical history",
    Document( "Roman Empire",
        page_content="A caloric deficit is scientifically required to lose body fat, regardless of the diet type.", 
        metadata={"category": "fitness", "id": "H2"}
    ),ey_features": ["law", "military", "architecture"]
    Document(
        page_content="Creatine monohydrate is a widely researched supplement that boosts physical performance.", 
        metadata={"category": "fitness", "id": "H3"}
    ), Document(
    Document(ent="The French Revolution began in 1789 and marked a turning point in European history. It led to the overthrow of the monarchy, the rise of republican ideals, and the spread of concepts such as liberty, equality, and fraternity across the world.",
        page_content="Consuming sufficient protein post-workout aids in muscle recovery and hypertrophy.", 
        metadata={"category": "fitness", "id": "H4"}
    ),vent": "French Revolution",
    "year": 1789,
    # --- Twitch Streaming Documents ---
    Document( "political transformation"
        page_content="OBS Studio is the most popular open-source software used for broadcasting on Twitch.", 
        metadata={"category": "streaming", "id": "T1"}
    ),
    Document(nt(
        page_content="Twitch chat sentiment can fluctuate rapidly depending on the streamer's gameplay performance.", s. It resulted in massive loss of life and led to significant geopolitical changes, including the formation of the United Nations and the beginning of the Cold War.",
        metadata={"category": "streaming", "id": "T2"}
    ),opic": "world history",
    Document("World War II",
        page_content="Raiding is a feature that allows a streamer to send their live viewers to another channel.", 
        metadata={"category": "streaming", "id": "T3"} War"]
    ),
    Document(
        page_content="Parasocial relationships often form when viewers feel a close personal bond with a streamer.", 
        metadata={"category": "streaming", "id": "T4"}
    ),ge_content="The Indian Independence Movement was a prolonged struggle against British colonial rule, culminating in independence in 1947. Leaders like Mahatma Gandhi promoted nonviolent resistance, which became a powerful method of political protest worldwide.",
]   metadata={
    "topic": "indian history",
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    "year": 1947,
vectorstore = FAISS.from_documents(documents=docs, embedding=embedding)
    "method": "nonviolent resistance"
}
simple_retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})

HISTORY_DOCUMENTS = [doc1,doc2,doc3,doc4,doc5]



