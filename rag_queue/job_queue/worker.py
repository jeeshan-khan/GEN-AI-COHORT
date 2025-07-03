# flake8: noqa
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore

from openai import OpenAI
import os
from dotenv import load_dotenv


load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")

client = OpenAI(
    api_key=api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    api_key=api_key
)

vector_db = QdrantVectorStore.from_existing_collection(
    url="http://vector-db:6333",
    collection_name="Learning_RAG",
    embedding=embeddings
)


async def process_query(query: str):
    print("Searching chunks", query),
    results = vector_db.similarity_search(
        query=query
    )
    context = "\n\n\n".join([
        f"Page Content: {result.page_content}\n Page Number: {result.metadata['page_label']}\nFile Location: {result.metadata['source']}"
        for result in results
    ])

    SYSTEM_PROMPT = f"""
    You are a helpful AI Assistant who answers user queries based on the
    available context retrieved from a PDF file along with page contents and
    page number.
    You should only answer the user based on the following context and guide
    them to open the right page number to know more.

    Context:
    {context}
    """
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": query}
    ]
    
    response = client.chat.completions.create(
        model="gemini-2.0-flash",
        messages=messages
    )
    
    # Save to DB
    print(f"🤖: {query}", response.choices[0].message.content, "\n\n\n")