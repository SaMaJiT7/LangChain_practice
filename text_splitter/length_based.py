from langchain_community.document_loaders import TextLoader,PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_text_splitters import CharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings


embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2') 


loader = PyPDFLoader('file.pdf')

docs = loader.load()

splitter = CharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=5
)

rec_splitter = RecursiveCharacterTextSplitter(
    chunk_size=80,
    chunk_overlap=5
)

semantic_splitter = SemanticChunker(
    embeddings,
    breakpoint_threshold_type="percentile" # type: ignore
)

texts = rec_splitter.split_documents(docs)

# Using SemanticChunker to split based on semantic similarity
embedded_texts = semantic_splitter.split_documents(docs)


# print(len(texts))

# print(texts[7].page_content)

print(len(embedded_texts))

print(embedded_texts[0].page_content)