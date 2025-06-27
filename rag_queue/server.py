from fastapi import FastAPI,Query
from .job_queue.connection import queue  # ✅ absolute import
from .job_queue.worker import process_query
app = FastAPI()

@app.get("/")
def chat():
    return {"status": "Server is up and running"}

@app.post("/chat")
def chat(
    query: str = Query(..., description="Chat Message"),
):
    #Query ko queue mein daaldo
    job_id = queue.enqueue(process_query, query) #process_query(query)

    #Aur phir user ko bolo your job received
    return {"status": "Your job has been received and queued","job_id":job_id.id}





