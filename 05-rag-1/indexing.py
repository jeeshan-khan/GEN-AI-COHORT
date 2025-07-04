from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from pathlib import Path
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")

# Load PDF
pdf_path = Path(__file__).parent / "example.pdf"
loader = PyPDFLoader(file_path=str(pdf_path))
docs = loader.load()
print(f"📄 Loaded {len(docs)} raw pages")

# Split into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=400
)
split_docs = text_splitter.split_documents(docs)
print(f"✂️ Split into {len(split_docs)} chunks")

# Create embeddings
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    api_key=api_key
)

# Save chunks to Qdrant
vector_store = QdrantVectorStore.from_documents(
    documents=split_docs,
    url="http://vector-db:6333",
    collection_name="Learning_RAG",
    embedding=embeddings
)

print("✅ Indexing of Documents Completed.")
