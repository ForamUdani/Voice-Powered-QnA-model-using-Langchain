# ingest.py

import os
from dotenv import load_dotenv
import time
from openai import RateLimitError

load_dotenv()

import pandas as pd
from datasets import load_dataset

from langchain_community.document_loaders import DataFrameLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

# --- Configuration ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
PINECONE_INDEX_NAME = "YOUR_PINECONE_INDEX_NAME" # Your chosen small index name

if not all([OPENAI_API_KEY, PINECONE_API_KEY, PINECONE_ENVIRONMENT]):
    raise ValueError("Please set OPENAI_API_KEY, PINECONE_API_KEY, and PINECONE_ENVIRONMENT environment variables.")

# --- Pinecone Client Initialization ---
print("Initializing Pinecone client...")
pc = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)

# --- Check if Index Exists and has Data ---
index_exists = PINECONE_INDEX_NAME in pc.list_indexes().names()
index_has_data = False
if index_exists:
    try:
        index_stats = pc.Index(PINECONE_INDEX_NAME).describe_index_stats()
        if index_stats and index_stats.total_vector_count > 0:
            index_has_data = True
            print(f"Pinecone index '{PINECONE_INDEX_NAME}' exists and contains {index_stats.total_vector_count} vectors.")
        else:
            print(f"Pinecone index '{PINECONE_INDEX_NAME}' exists but is empty.")
    except Exception as e:
        print(f"Could not get index stats for '{PINECONE_INDEX_NAME}', assuming it might be empty or problematic: {e}")
        index_exists = False

if index_exists and index_has_data:
    print(f"Skipping data ingestion as index '{PINECONE_INDEX_NAME}' already contains data.")
    print("If you need to re-ingest, please delete the index from your Pinecone dashboard first,")
    print("or change PINECONE_INDEX_NAME in your .env file.")
else:  
    print(f"Pinecone index '{PINECONE_INDEX_NAME}' is either new or empty. Proceeding with ingestion.")

    # --- Step 1: Load and Preprocess Dataset (100-Row Subset) ---
    print("Loading original dataset and taking first 100 rows...")
    dataset = load_dataset("jamescalam/ai-arxiv-chunked", split="train")
    df = dataset.to_pandas()
    df = df.head(100) # Take only the first 100 rows
    print(f"Processing a subset of {len(df)} rows from the original dataset.")

    df = df.dropna(subset=["chunk"]) # Still drop NaNs if any in the subset

    # Metadata processing
    df['publication_year'] = df['published'].astype(str).str[:4]
    df['first_author'] = df['authors'].apply(
        lambda x: x[0]['name'] if isinstance(x, list) and len(x) > 0 and 'name' in x[0] else 'Unknown'
    )

    # --- CRITICAL CHANGE: DROP ALL POTENTIALLY PROBLEMATIC/UNUSED ORIGINAL COLUMNS FROM METADATA ---
    # These columns often contain complex types (like lists that Pandas treats as objects/ndarrays)
    # or problematic 'null' values that Pinecone's metadata does not directly support.
    # Added 'comment' to the list.
    problematic_original_columns = ['authors', 'categories', 'references', 'id', 'published', 'url', 'journal_ref', 'comment']
    columns_to_drop_from_df = [col for col in problematic_original_columns if col in df.columns]

    if columns_to_drop_from_df:
        df = df.drop(columns=columns_to_drop_from_df) 
        print(f"Dropped problematic/unused original columns from DataFrame: {', '.join(columns_to_drop_from_df)}.")
    # --- END CRITICAL CHANGE ---

    # Combine 'title' and 'chunk' into a single 'page_content' column for LangChain Document
    if 'abstract' in df.columns:
        df["page_content"] = df.apply(lambda row: f"Title: {row['title']}\n\nAbstract: {row['abstract']}\n\nChunk: {row['chunk']}", axis=1)
    else:
        df["page_content"] = df.apply(lambda row: f"Title: {row['title']}\n\nChunk: {row['chunk']}", axis=1)


    loader = DataFrameLoader(df, page_content_column="page_content")
    documents = loader.load()
    print(f"Loaded {len(documents)} documents for processing.")
    print("Example metadata for first document after dropping columns:")
    if documents:
        print(documents[0].metadata)

    # --- Step 2: Preprocess and Chunk Data ---
    print("Chunking documents...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks after re-splitting.")

    # --- Step 3: Embed Chunks and Store in Pinecone ---
    print("Initializing OpenAI Embeddings...")
    embeddings_model = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

    if not index_exists:
        print(f"Creating Pinecone index: {PINECONE_INDEX_NAME}...")
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=embeddings_model.model_dim,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        print(f"Pinecone index '{PINECONE_INDEX_NAME}' created.")
    else:
        print(f"Pinecone index '{PINECONE_INDEX_NAME}' already exists but was empty. Populating now.")

    print("Starting batched embedding and upsert to Pinecone (with rate limit handling)...")
    pinecone_index = pc.Index(PINECONE_INDEX_NAME)

    batch_size = 100
    sleep_time = 10

    for i in range(0, len(chunks), batch_size):
        batch_chunks = chunks[i:i + batch_size]
        texts = [chunk.page_content for chunk in batch_chunks]
        metadatas = [chunk.metadata for chunk in batch_chunks]

        retries = 0
        max_retries = 5
        base_delay = 5

        while retries < max_retries:
            try:
                embeddings_list = embeddings_model.embed_documents(texts)

                vectors_to_upsert = []
                for j, chunk_text in enumerate(texts):
                    unique_id = f"doc_{i+j}_chunk_{hash(chunk_text) % 1000000}"
                    vectors_to_upsert.append((unique_id, embeddings_list[j], metadatas[j]))

                pinecone_index.upsert(vectors=vectors_to_upsert)

                print(f"Upserted batch {i // batch_size + 1}/{(len(chunks) + batch_size - 1) // batch_size} ({len(batch_chunks)} chunks).")
                time.sleep(sleep_time)
                break

            except RateLimitError as e:
                retries += 1
                delay = base_delay * (2 ** (retries - 1))
                print(f"Rate limit hit. Retrying in {delay} seconds (Attempt {retries}/{max_retries})... Error: {e}")
                time.sleep(delay)
            except Exception as e:
                print(f"An error occurred during embedding/upsert for batch {i // batch_size + 1}: {e}")
                retries += 1
                delay = base_delay * (2 ** (retries - 1))
                print(f"Retrying in {delay} seconds (Attempt {retries}/{max_retries})...")
                time.sleep(delay)

        if retries == max_retries:
            print(f"Failed to process batch {i // batch_size + 1} after {max_retries} retries. Skipping remaining chunks.")
            break

    print("Batched embedding and upsert process completed.")

print("Ingestion process completed.")
