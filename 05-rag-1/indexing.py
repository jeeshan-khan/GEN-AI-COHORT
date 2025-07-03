from langchain_community.document_loaders import PyPDFLoader
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore
import os
from dotenv import load_dotenv


load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")

pdf_path = Path(__file__).parent / "example.pdf"
loader = PyPDFLoader(file_path=pdf_path)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=400
)

split_docs = text_splitter.split_documents(documents=docs)

# embedding model
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    api_key=api_key
)


# vector store

vector_store = QdrantVectorStore.from_documents(
    documents=split_docs,
    url="http://vector-db:6333",
    collection_name="Learning_RAG",
    embedding=embeddings
)

print("Indexing of Documents Completed......")
