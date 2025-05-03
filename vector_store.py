import os
import uuid
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import numpy as np

# Manual fallback for env variables if .env is not used
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY") or "your_pinecone_api_key"
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME") or "your_index_name"

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

# Create index if not exists
if PINECONE_INDEX_NAME not in [index.name for index in pc.list_indexes()]:
    pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

index = pc.Index(PINECONE_INDEX_NAME)

# Embedding and QA models
model = SentenceTransformer("all-MiniLM-L6-v2")  # Embedding model for sentence encoding
qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")  # QA model

# Function to chunk text into smaller parts (optimal size for understanding context)
def chunk_text(text, chunk_size=300):
    if chunk_size <= 0:
        raise ValueError("Chunk size must be a positive integer.")
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

# Function to embed and store document chunks in Pinecone
def embed_and_store(docs):
    if isinstance(docs, str):
        docs = [docs]
    embeddings = model.encode(docs).tolist()
    vectors = [{"id": str(uuid.uuid4()), "values": emb, "metadata": {"text": doc}} for doc, emb in zip(docs, embeddings)]
    index.upsert(vectors=vectors)

# Function to query Pinecone and get the most relevant document chunks based on query
def query_pinecone(query_text, top_k=3):
    query_vector = model.encode([query_text])[0].tolist()
    result = index.query(vector=query_vector, top_k=top_k, include_metadata=True)
    return [match["metadata"]["text"] for match in result.get("matches", [])]

# Function to extract the best answer from the returned chunks using the QA pipeline
def get_answer_from_chunks(question, chunks, min_score_threshold=0.8):
    best_answer = ""
    best_score = 0
    relevant_chunks = []
    
    # Go through all chunks and get answers
    for chunk in chunks:
        try:
            result = qa_pipeline(question=question, context=chunk)  # Get QA result from chunk
            if result["score"] > min_score_threshold:  # Filter out answers with a low score
                relevant_chunks.append((result["score"], result["answer"]))  # Store answer with score
        except Exception as e:
            print(f"Error processing chunk: {e}")
            continue
    
    # Select the best answer based on the highest score
    if relevant_chunks:
        best_score, best_answer = max(relevant_chunks, key=lambda x: x[0])
    else:
        best_answer = "Sorry, I couldn't find a specific answer."
    
    return best_answer

# Function to perform the complete process: chunk text, embed, store, query, and get answer
def process_pdf_query(pdf_text, question):
    chunks = chunk_text(pdf_text)  # Split the document into manageable chunks
    embed_and_store(chunks)  # Embed and store in Pinecone

    # Query Pinecone for relevant document chunks
    relevant_chunks = query_pinecone(question)  # Get most relevant chunks

    # Extract the best answer from the relevant chunks
    answer = get_answer_from_chunks(question, relevant_chunks)
    return answer






