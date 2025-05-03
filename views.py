import hashlib
import os
import uuid
from django.shortcuts import render
from django.http import JsonResponse
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
from django.views.decorators.csrf import csrf_exempt

# Load environment variables
load_dotenv()

# Initialize Pinecone
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
pc = Pinecone(api_key=PINECONE_API_KEY)

# Create index if it doesn't exist
if PINECONE_INDEX_NAME not in [index.name for index in pc.list_indexes()]:
    pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

index = pc.Index(PINECONE_INDEX_NAME)
model = SentenceTransformer("all-MiniLM-L6-v2")

# Function to generate a hash for the uploaded PDF
def get_pdf_hash(file):
    file.seek(0)
    content = file.read()
    file.seek(0)
    return hashlib.md5(content).hexdigest()

# Function to chunk text into smaller parts
def chunk_text(text, chunk_size=300, overlap=50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk:
            chunks.append(chunk)
    return chunks

# Function to embed and store text chunks in Pinecone
def embed_and_store(chunks, file_id):
    embeddings = model.encode(chunks).tolist()
    vectors = [
        {
            "id": str(uuid.uuid4()),
            "values": emb,
            "metadata": {"text": chunk, "file_id": file_id}
        }
        for chunk, emb in zip(chunks, embeddings)
    ]
    index.upsert(vectors=vectors)

# Function to delete all previous chunks in Pinecone
def delete_all_previous_chunks():
    # Deletes all vectors in the Pinecone index
    index.delete(delete_all=True)

# Function to query Pinecone and return the most relevant answer
def query_pinecone(query_text, previous_chunks, top_k=10):
    query_vector = model.encode([query_text])[0].tolist()
    result = index.query(vector=query_vector, top_k=top_k, include_metadata=True)
    seen = set(previous_chunks)
    for match in result["matches"]:
        text = match["metadata"]["text"]
        if text not in seen:
            return text  # Return the most relevant, unseen match
    return "No new relevant information found."

# CSRF exemption for POST request (only needed for testing in a local environment)
@csrf_exempt
def chat_bot_view(request):
    if request.method == 'POST':
        uploaded_file = request.FILES.get('pdf_file')
        user_query = request.POST.get('user_query')

        # Initialize session memory if it doesn't exist
        if 'memory' not in request.session:
            request.session['memory'] = []

        memory = set(request.session['memory'])

        if uploaded_file:
            try:
                # Delete all previous chunks from Pinecone when a new file is uploaded
                delete_all_previous_chunks()

                # Process the new PDF file
                file_hash = get_pdf_hash(uploaded_file)
                pdf = PdfReader(uploaded_file)
                full_text = "".join(page.extract_text() or "" for page in pdf.pages)
                chunks = chunk_text(full_text)
                embed_and_store(chunks, file_id=file_hash)

                # Reset memory for the new file
                memory.clear()
            except Exception as e:
                return JsonResponse({"error": f"Failed to process PDF: {str(e)}"}, status=400)

        if user_query:
            try:
                # Query Pinecone and return the most relevant answer
                answer = query_pinecone(user_query, memory)
                memory.add(answer)  # Add the answer to the session memory
                request.session['memory'] = list(memory)  # Update session memory
                return JsonResponse({"answer": answer})
            except Exception as e:
                return JsonResponse({"error": f"Query failed: {str(e)}"}, status=500)

        return JsonResponse({"error": "No query provided."}, status=400)

    return render(request, 'chatbot.html')
