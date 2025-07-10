# flake8: noqa
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from pathlib import Path
from dotenv import load_dotenv
import os
import requests

# Load environment variables
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

try:
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
    collection_name = "Learning_RAG"
    vector_store = QdrantVectorStore.from_documents(
        documents=split_docs,
        url="http://vector-db:6333",
        collection_name=collection_name,
        embedding=embeddings
    )

    print("✅ Indexing of Documents Completed.")
    print(f"📁 Documents stored in collection: {collection_name}")

    # Verify with Qdrant API
    
    try:
        res = requests.get("http://vector-db:6333/collections")
        res.raise_for_status()
        data = res.json()

        print("📚 Collections available in Qdrant:")
        if "result" in data and isinstance(data["result"], list):
            for col in data["result"]:
                if isinstance(col, dict) and "name" in col:
                    print(f"  🔸 {col['name']}")
                else:
                    print(f"  ⚠️ Unexpected collection format: {col}")
        else:
            print("⚠️ Unexpected response format from Qdrant:")
            print(data)

    except Exception as fetch_err:
        print("⚠️ Could not fetch collection list from Qdrant.")
        print(f"Fetch error: {fetch_err}")


except Exception as e:
    print("❌ Failed to index documents.")
    print(f"Error: {e}")
