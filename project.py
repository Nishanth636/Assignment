import os
import numpy as np
import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import streamlit as st
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
INDEX_NAME = "project"

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

# Ensure the index exists
if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME, 
        dimension=384, 
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

# Connect to the index
index = pc.Index(INDEX_NAME)

# Load embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Load QA pipeline
qa_pipeline = pipeline(
    "question-answering", 
    model="distilbert-base-cased-distilled-squad",
    use_auth_token=HUGGINGFACE_API_KEY
)

def scrape_website(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        paragraphs = soup.find_all('p')
        content = ' '.join([para.get_text() for para in paragraphs])
        if not content.strip():
            st.error("⚠️ No content found! Check the website structure.")
        return content
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching website content: {e}")
        return ""

def embed_and_store_texts(texts):
    embeddings = embedding_model.encode(texts).tolist()
    ids = [str(i) for i in range(len(texts))]
    vectors = list(zip(ids, embeddings, [{'text': text} for text in texts]))
    index.upsert(vectors=vectors)
    st.success("✅ Content stored successfully!")

def retrieve_similar_text(query):
    query_embedding = embedding_model.encode([query]).flatten().tolist()
    results = index.query(query_embedding, top_k=5, include_metadata=True)
    retrieved_texts = [match['metadata']['text'] for match in results['matches']]
    return retrieved_texts

def get_answer_from_context(question, context):
    if not context:
        return "No relevant content found in database!"
    result = qa_pipeline(question=question, context=context)
    return result['answer']

# Streamlit UI
st.title("AI-Powered Web Scraper & QA Bot")
url = st.text_input("Enter a website URL to scrape:", "https://en.wikipedia.org/wiki/Artificial_intelligence")

if url:
    st.write("🔄 Scraping content...")
    website_content = scrape_website(url)
    
    if website_content:
        st.write("✅ Content scraped successfully!")
        chunks = website_content.split('\n')
        st.write("🗄 Storing content in Pinecone...")
        embed_and_store_texts(chunks)
        user_query = st.text_input("💬 Ask a question based on the scraped content:")
        
        if user_query:
            st.write("🔍 Retrieving relevant information...")
            retrieved_texts = retrieve_similar_text(user_query)
            context = " ".join(retrieved_texts)
            st.write("🤖 Generating answer...")
            answer = get_answer_from_context(user_query, context)
            st.write(f"📢 Answer: {answer}")
