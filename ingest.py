# ingest.py

import os
import pandas as pd
from datasets import load_dataset
from dotenv import load_dotenv

from langchain_community.document_loaders import DataFrameLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

# Load environment variables from .env file
load_dotenv()

# --- Configuration ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT") # e.g., "gcp-starter" or your specific environment
PINECONE_INDEX_NAME = "ai-arxiv-chunks-langchain" # Name for your Pinecone index

if not all([OPENAI_API_KEY, PINECONE_API_KEY, PINECONE_ENVIRONMENT]):
    raise ValueError("Please set OPENAI_API_KEY, PINECONE_API_KEY, and PINECONE_ENVIRONMENT environment variables.")

# --- Step 1: Load and Preprocess Dataset ---
print("Loading dataset...")
dataset = load_dataset("jamescalam/ai-arxiv-chunked", split="train")
df = dataset.to_pandas()

# Drop rows where 'chunk' is NaN, as these cannot be processed
df = df.dropna(subset=["chunk"])

# Combine 'title' and 'chunk' into a single 'page_content' column for LangChain Document
# Also, include other relevant metadata
df["page_content"] = df.apply(lambda row: f"Title: {row['title']}\n\nChunk: {row['chunk']}", axis=1)

# Create LangChain Documents from the DataFrame
# The 'page_content_column' specifies which DataFrame column holds the main text.
# All other columns will be treated as metadata.
loader = DataFrameLoader(df, page_content_column="page_content")
documents = loader.load()

print(f"Loaded {len(documents)} documents.")
print("First document example:")
print(documents[0].page_content)
print("First document metadata example:")
print(documents[0].metadata)

# --- Step 2: Preprocess and Chunk Data ---
print("Chunking documents...")
# Using RecursiveCharacterTextSplitter for robust text splitting
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,      # Max size of each chunk
    chunk_overlap=200,    # Overlap between chunks to maintain context
    length_function=len,  # How length is measured (characters)
    is_separator_regex=False, # Treat separators as plain strings
)

chunks = text_splitter.split_documents(documents)
print(f"Created {len(chunks)} chunks.")
print("First chunk example:")
print(chunks[0].page_content)
print("First chunk metadata example:")
print(chunks[0].metadata)


# --- Step 3: Embed Chunks and Store in Pinecone ---
print("Initializing OpenAI Embeddings...")
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

print("Initializing Pinecone...")
# Initialize Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)

# Check if the index already exists, if not, create it
if PINECONE_INDEX_NAME not in pc.list_indexes().names():
    print(f"Creating Pinecone index: {PINECONE_INDEX_NAME}...")
    pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=embeddings.model_dim,  # OpenAI embeddings have 1536 dimensions
        metric="cosine",                 # Cosine similarity is common for embeddings
        spec=ServerlessSpec(cloud="aws", region="us-east-1") # Example serverless spec, adjust as needed
    )
    print(f"Pinecone index '{PINECONE_INDEX_NAME}' created.")
else:
    print(f"Pinecone index '{PINECONE_INDEX_NAME}' already exists.")

print("Adding documents to Pinecone (this may take a while)...")
# Use PineconeVectorStore.from_documents to embed and upload documents
# This will automatically create the index if it doesn't exist (though we explicitly check above)
# and handle batching.
vectorstore = PineconeVectorStore.from_documents(
    documents=chunks,
    embedding=embeddings,
    index_name=PINECONE_INDEX_NAME
)
print("Documents successfully added to Pinecone!")
print(f"You can now query your index '{PINECONE_INDEX_NAME}' in Pinecone.")

