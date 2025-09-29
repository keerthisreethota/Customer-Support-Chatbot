# ingest.py
import os
import uuid
import asyncio
import aiohttp
import backoff
import time
import json
from typing import List
from datetime import datetime
from dotenv import load_dotenv

from langchain_community.vectorstores import AzureSearch
from langchain_community.document_loaders import (
    DirectoryLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredPDFLoader,
    TextLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

# Loading Environment Variables
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

#Custom Embedding Function
@backoff.on_exception(backoff.expo, Exception, max_tries=5)
async def get_embedding(session, text):
    headers = {"api-key": EMBEDDING_API_KEY, "Content-Type": "application/json"}
    payload = {"model": EMBEDDING_MODEL, "input": text}
    async with session.post(EMBEDDING_API_URL, headers=headers, json=payload) as resp:
        if resp.status != 200:
            text_resp = await resp.text()
            raise Exception(f"Embedding API error: {resp.status}, {text_resp}")
        data = await resp.json()
        return data["data"][0]["embedding"]

async def get_embeddings_batch(texts):
    async with aiohttp.ClientSession() as session:
        tasks = [get_embedding(session, t) for t in texts]
        return await asyncio.gather(*tasks)

def embedding_function_sync(texts: List[str]) -> List[List[float]]:
    """Sync wrapper for embedding function that returns List[List[float]]"""
    print(f" Generating embeddings for {len(texts)} texts...")
    embeddings = asyncio.run(get_embeddings_batch(texts))
    print(f" Generated {len(embeddings)} embeddings")
    return embeddings

#Loading Documents
def load_documents():
    print(" Loading documents from ./docs folder...")
    if not os.path.exists("./docs"):
        print(" ./docs folder does not exist!")
        return []

    loaders = {
        "*.docx": UnstructuredWordDocumentLoader,
        "*.pdf": UnstructuredPDFLoader,
        "*.txt": TextLoader
    }

    all_docs = []
    for pattern, loader_cls in loaders.items():
        try:
            loader = DirectoryLoader("./docs", glob=pattern, loader_cls=loader_cls)
            docs = loader.load()
            print(f"Loaded {len(docs)} {pattern} documents.")
            all_docs.extend(docs)
        except Exception as e:
            print(f"Warning: Failed to load {pattern} files: {e}")

    print(f"Total documents loaded: {len(all_docs)}")
    return all_docs

#Splitting Documents
def split_documents(documents):
    if not documents:
        print("No documents to split.")
        return []

    print("Splitting documents into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = splitter.split_documents(documents)
    print(f" Created {len(chunks)} chunks.")
    return chunks

#Preparing Chunks
def prepare_chunks(chunks):
    """Convert LangChain chunks to match Azure Index schema exactly."""
    prepared = []
    for i, doc in enumerate(chunks):
        metadata = doc.metadata or {}
        source = metadata.get("source", "")
        
        # Clean source path for better readability
        if source:
            source = os.path.basename(source)

        # For Extracting potential question/answer pairs from content
        content = doc.page_content.strip()
        question, answer = extract_qa_from_content(content)
        
        # Get current timestamp in ISO format
        current_timestamp = datetime.now().isoformat()

        prepared.append(
            Document(
                page_content=content,
                metadata={
                    "id": str(uuid.uuid4()),
                    "question": question,
                    "answer": answer,
                    "source": source,
                    "category": metadata.get("category", "general"),
                    "section": metadata.get("section", ""),
                    "timestamp": current_timestamp,
                    "numeric_value": float(i)
                }
            )
        )
    return prepared

def extract_qa_from_content(content):
    """Extract potential question and answer from content."""
    lines = content.split('\n')
    question = ""
    answer = content
    
    for i, line in enumerate(lines):
        line = line.strip()
        if line.startswith(('Q:', 'Q', 'Question:', 'QUESTION:', '?')) and len(line) > 10:
            question = line
            if i + 1 < len(lines):
                answer = lines[i + 1].strip()
            break
        elif line.endswith('?') and len(line) > 5:
            question = line
            if i + 1 < len(lines):
                answer = lines[i + 1].strip()
            break
    
    return question, answer

#Alternative: Direct Azure Search Client
def index_with_direct_client(chunks):
    """Use Azure Search client directly to avoid LangChain compatibility issues"""
    try:
        from azure.search.documents import SearchClient
        from azure.core.credentials import AzureKeyCredential
        from azure.search.documents.indexes import SearchIndexClient
        from azure.search.documents.models import VectorizedQuery
        
        print(" Using direct Azure Search client...")
        
        # Creating search client
        credential = AzureKeyCredential(AZURE_SEARCH_KEY)
        search_client = SearchClient(
            endpoint=AZURE_SEARCH_ENDPOINT,
            index_name=AZURE_SEARCH_INDEX,
            credential=credential
        )
        
        docs_to_index = prepare_chunks(chunks)
        
        # Converting to Azure Search documents
        azure_documents = []
        for doc in docs_to_index:
            #To Generate embedding for each document
            embedding = embedding_function_sync([doc.page_content])[0]
            
            azure_doc = {
                "id": doc.metadata["id"],
                "question": doc.metadata["question"],
                "answer": doc.metadata["answer"],
                "content": doc.page_content,
                "content_vector": embedding,
                "source": doc.metadata["source"],
                "category": doc.metadata["category"],
                "section": doc.metadata["section"],
                "timestamp": doc.metadata["timestamp"],
                "numeric_value": doc.metadata["numeric_value"]
            }
            azure_documents.append(azure_doc)
        
        # Upload documents in batches
        batch_size = 10
        total_uploaded = 0
        
        for i in range(0, len(azure_documents), batch_size):
            batch = azure_documents[i:i + batch_size]
            try:
                result = search_client.upload_documents(batch)
                succeeded = [r for r in result if r.succeeded]
                total_uploaded += len(succeeded)
                print(f" Uploaded batch {i//batch_size + 1}: {len(succeeded)}/{len(batch)} documents")
                
                if len(succeeded) < len(batch):
                    failed = [r for r in result if not r.succeeded]
                    for f in failed:
                        print(f" Failed: {f.key} - {f.error}")
                        
            except Exception as e:
                print(f" Batch upload failed: {e}")
        
        print(f" Total documents uploaded: {total_uploaded}/{len(azure_documents)}")
        return total_uploaded
        
    except ImportError:
        print(" Azure Search SDK not installed. Install with: pip install azure-search-documents")
        return 0
    except Exception as e:
        print(f" Direct client approach failed: {e}")
        return 0

#Indexing Documents
def index_documents(chunks, batch_size=5):
    if not chunks:
        print(" No chunks to index.")
        return
        
    if not AZURE_SEARCH_ENDPOINT or not AZURE_SEARCH_KEY or not AZURE_SEARCH_INDEX:
        print(" Azure Search environment variables missing!")
        return

    docs_to_index = prepare_chunks(chunks)

    print(" Sample document structure:")
    sample_doc = docs_to_index[0]
    print(f"ID: {sample_doc.metadata['id']}")
    print(f"Question: {sample_doc.metadata['question']}")
    print(f"Answer preview: {sample_doc.metadata['answer'][:100]}...")
    print(f"Content preview: {sample_doc.page_content[:100]}...")

    print(" Attempting to index documents...")
    
    # First try the direct client approach
    print("\n Trying direct Azure Search client approach...")
    uploaded_count = index_with_direct_client(chunks)
    
    if uploaded_count > 0:
        print(f" Successfully uploaded {uploaded_count} documents using direct client!")
        return
    
    # Fallback to LangChain approach with error handling
    print("\n Falling back to LangChain approach...")
    try:
        # Try with minimal configuration
        vector_store = AzureSearch(
            azure_search_endpoint=AZURE_SEARCH_ENDPOINT,
            azure_search_key=AZURE_SEARCH_KEY,
            index_name=AZURE_SEARCH_INDEX,
            embedding_function=embedding_function_sync,
        )

        # Trying single document first to test
        print(" Testing with single document...")
        try:
            vector_store.add_documents([docs_to_index[0]])
            print(" Single document test successful! Proceeding with batch upload...")
            
            for i in range(0, len(docs_to_index), batch_size):
                batch = docs_to_index[i:i + batch_size]
                batch_num = (i // batch_size) + 1
                
                try:
                    print(f" Processing batch {batch_num}...")
                    vector_store.add_documents(batch)
                    print(f" Successfully indexed batch {batch_num}")
                    
                except Exception as e:
                    print(f" Batch {batch_num} failed: {e}")
                    
        except Exception as e:
            print(f" Single document test failed: {e}")
            print(" Try updating your packages: pip install --upgrade langchain-community azure-search-documents")
                        
    except Exception as e:
        print(f" All indexing approaches failed: {e}")
        print(" Recommendations:")
        print("1. Update packages: pip install --upgrade langchain-community azure-search-documents")
        print("2. Check your Azure Search index configuration")
        print("3. Verify your embedding API is working correctly")

#GPT-4 Summary
def chat_with_gpt4(prompt):
    import requests
    headers = {"api-key": LLM_API_KEY, "Content-Type": "application/json"}
    payload = {"model": LLM_MODEL, "messages": [{"role": "user", "content": prompt}]}
    response = requests.post(LLM_API_URL, headers=headers, json=payload)
    if response.status_code != 200:
        raise Exception(f"Chat API error: {response.status_code}, {response.text}")
    return response.json()["choices"][0]["message"]["content"]

#Main
def main():
    print(" Starting document ingestion process...")
    
    # Validating environment variables
    required_vars = [
        "AZURE_SEARCH_ENDPOINT", "AZURE_SEARCH_KEY", "AZURE_SEARCH_INDEX",
        "EMBEDDING_API_URL", "EMBEDDING_API_KEY", "EMBEDDING_MODEL"
    ]
    
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        print(f" Missing environment variables: {', '.join(missing_vars)}")
        return

    documents = load_documents()
    if not documents:
        print(" No documents found to process.")
        return

    chunks = split_documents(documents)
    if not chunks:
        print(" No chunks created from documents.")
        return

    print(f" Starting indexing of {len(chunks)} chunks...")
    index_documents(chunks, batch_size=3)

    #GPT-4 summary
    try:
        print("\n Generating document summary with GPT-4...")
        summary_prompt = f"Summarize the {len(chunks)} document chunks about customer support and FAQs in 2-3 sentences."
        summary = chat_with_gpt4(summary_prompt)
        print("\n GPT-4 Summary:")
        print(summary)
    except Exception as e:
        print(f" GPT-4 summary failed: {e}")

    print("Document ingestion process completed!")

if __name__ == "__main__":
    main()