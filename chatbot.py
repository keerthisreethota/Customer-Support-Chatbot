import os
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
from azure.core.credentials import AzureKeyCredential


# Load environment variables

load_dotenv()

AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
AZURE_SEARCH_KEY = os.getenv("AZURE_SEARCH_KEY")
AZURE_SEARCH_INDEX = os.getenv("AZURE_SEARCH_INDEX")

EMBEDDING_API_URL = os.getenv("EMBEDDING_API_URL")
EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")

LLM_API_URL = os.getenv("LLM_API_URL")
LLM_API_KEY = os.getenv("LLM_API_KEY")
LLM_MODEL = os.getenv("LLM_MODEL")

# Initialize Search Client
search_client = SearchClient(
    endpoint=AZURE_SEARCH_ENDPOINT,
    index_name=AZURE_SEARCH_INDEX,
    credential=AzureKeyCredential(AZURE_SEARCH_KEY)
)

#  FastAPI Setup

app = FastAPI(title="Customer Support Chatbot", docs_url="/") # docs_url="/" is the new path for Swagger UI

class QueryRequest(BaseModel):
    query: str
    top_k: int = 3

class QueryResponse(BaseModel):
    answer: str
    sources: list

#  Helper Functions

def generate_embedding(text: str):
    """Generate embedding using Azure AI Foundry REST API."""
    headers = {
        "Content-Type": "application/json",
        "api-key": EMBEDDING_API_KEY
    }
    body = {"input": text}
    response = requests.post(EMBEDDING_API_URL, headers=headers, json=body)
    if response.status_code != 200:
        raise Exception(f"Embedding API error: {response.text}")
    data = response.json()
    return data["data"][0]["embedding"]

def retrieve_documents(query: str, top_k: int = 3):
    """Retrieve top documents from Azure Cognitive Search."""
    vector = generate_embedding(query)
    
    vector_query = VectorizedQuery(
        vector=vector,
        k_nearest_neighbors=top_k,
        fields="content_vector"
    )
    
    results = search_client.search(
        search_text="",  # empty since vector search
        vector_queries=[vector_query]
    )
    
    docs = []
    for result in results:
        docs.append({
            "content": result["content"],
            "source": result["source"]
        })
    return docs


def generate_answer(query: str, docs: list):
    """Generate answer using Azure AI Foundry Chat Completion API."""
    context = "\n\n".join([d["content"] for d in docs])
    prompt = f"""
    You are a helpful customer support assistant.
    Use the following context to answer the user query.

    Context:
    {context}

    Question: {query}
    Answer:
    """

    headers = {
        "Content-Type": "application/json",
        "api-key": LLM_API_KEY
    }
    body = {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 500,
        "temperature": 0,
        "model": LLM_MODEL
    }
    response = requests.post(LLM_API_URL, headers=headers, json=body)
    if response.status_code != 200:
        raise Exception(f"LLM API error: {response.text}")
    data = response.json()
    return data["choices"][0]["message"]["content"].strip()


#  FastAPI Endpoints

@app.post("/chat", response_model=QueryResponse)
async def chat(request: QueryRequest):
    try:
        docs = retrieve_documents(request.query, request.top_k)
        if not docs:
            return QueryResponse(answer="Sorry, I couldn't find relevant information.", sources=[])
        
        answer = generate_answer(request.query, docs)
        sources = [doc["source"] for doc in docs]
        
        return QueryResponse(answer=answer, sources=sources)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    return {"status": "ok"}
