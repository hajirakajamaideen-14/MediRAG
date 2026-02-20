import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings

DATA_PATH = "data"

documents = []

for file in os.listdir(DATA_PATH):
    if file.endswith(".pdf"):
        loader = PyPDFLoader(os.path.join(DATA_PATH, file))
        documents.extend(loader.load())

print(f"Loaded {len(documents)} pages.")

# Split into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150
)

chunks = text_splitter.split_documents(documents)

print(f"Total chunks created: {len(chunks)}")

# Create embeddings
embedding_model = OllamaEmbeddings(model="nomic-embed-text")

vectorstore = FAISS.from_documents(chunks, embedding_model)

vectorstore.save_local("faiss_index")

print("FAISS index created successfully.")

