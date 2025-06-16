
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore
import os
from openai import OpenAI

client = OpenAI(
    api_key="AIzaSyCRgdqi8elWuIGBpr-ecC7lecM_0q2kwr4",
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

os.environ["GOOGLE_API_KEY"] = "AIzaSyCRgdqi8elWuIGBpr-ecC7lecM_0q2kwr4"

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    api_key="AIzaSyCRgdqi8elWuIGBpr-ecC7lecM_0q2kwr4"
)

vector_db = QdrantVectorStore.from_existing_collection(
    url="http://localhost:6333",
    collection_name="Learning_RAG",
    embedding=embeddings
)

print("🤖 Ask me anything! (type 'exit' to quit)\n")

while True:
    query = input("Query: ")

    if query.lower() in ["exit", "quit"]:
        print("Goodbye!")
        break

    results = vector_db.similarity_search(query=query)

    context = "\n\n\n".join([
        f"Page Content: {result.page_content}\nPage Number: {result.metadata['page_label']}\nFile Location: {result.metadata['source']}"
        for result in results
    ])

    SYSTEM_PROMPT = f"""
    You are a helpful AI Assistant who answers user queries based on the available context
    retrieved from a PDF file along with page contents and page number.

    You should only answer the user based on the following context and guide them
    to open the right page number to know more.

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

    print(f"\n🤖: {response.choices[0].message.content}\n")
