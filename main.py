# main.py

import os
from dotenv import load_dotenv

load_dotenv()

from typing import List, Tuple, Dict, Any, Optional
import uuid
import datetime
import json
import asyncio # Import asyncio for async operations

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain.chains import RetrievalQA
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_community.document_loaders import DataFrameLoader
from datasets import load_dataset
import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter


import firebase_admin
from firebase_admin import credentials, firestore


# --- Configuration ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
PINECONE_INDEX_NAME = "YOUR_PINECONE_INDEX_NAME" # Your chosen small index name

if not all([OPENAI_API_KEY, PINECONE_API_KEY, PINECONE_ENVIRONMENT]):
    raise ValueError("Please set OPENAI_API_KEY, PINECONE_API_KEY, and PINECONE_ENVIRONMENT environment variables.")

# --- Firebase Initialization ---
db = None
app_id = "default-app-id" # Default fallback for local testing without Canvas

try:
    # Check for Canvas-specific Firebase config injected by the environment
    if '__firebase_config' in globals() and '__app_id' in globals():
        firebase_config_dict = json.loads(globals().get('__firebase_config'))
        app_id = globals().get('__app_id')

        cred = credentials.Certificate(firebase_config_dict)
        if not firebase_admin._apps: # Initialize only if no app is already initialized
            firebase_admin.initialize_app(cred)
        db = firestore.client()
        print(f"Firebase initialized for Canvas app ID: {app_id}")
    else:
        # Fallback for local development using serviceAccountKey.json
        cred_path = os.path.join(os.path.dirname(__file__), "serviceAccountKey.json")
        if os.path.exists(cred_path):
            cred = credentials.Certificate(cred_path)
            if not firebase_admin._apps:
                firebase_admin.initialize_app(cred)
            db = firestore.client()
            print("Firebase initialized using serviceAccountKey.json for local development.")
        else:
            print(f"WARNING: serviceAccountKey.json not found at {cred_path}. Feedback functionality might be limited.")
            db = None
except Exception as e:
    print(f"Error initializing Firebase: {e}")
    db = None

if db is None:
    print("CRITICAL: Firestore database not initialized. Feedback functionality will NOT work.")


# --- LangChain Setup ---
print("Initializing LangChain components...")

# 1. Embeddings
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# 2. Pinecone Vector Store
pinecone_vectorstore = PineconeVectorStore(
    index_name=PINECONE_INDEX_NAME,
    embedding=embeddings
)
print(f"Pinecone vectorstore initialized for index: {PINECONE_INDEX_NAME}")

# 3. BM25 Retriever (for Hybrid Search)
print("Loading dataset for BM25 indexing...")
bm25_dataset_full = load_dataset("jamescalam/ai-arxiv-chunked", split="train")
bm25_df = bm25_dataset_full.to_pandas().head(100) # Take first 100 rows for BM25

bm25_df = bm25_df.dropna(subset=["chunk"])
bm25_df["page_content"] = bm25_df.apply(lambda row: f"Title: {row['title']}\n\nChunk: {row['chunk']}", axis=1)

# Ensure BM25 documents have valid page_content before creating BM25Retriever
bm25_df = bm25_df[bm25_df['page_content'].str.strip().astype(bool)]
print(f"BM25 DataFrame has {len(bm25_df)} rows after dropping NaNs and empty page_content.")

bm25_text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
    is_separator_regex=False,
)
bm25_documents = bm25_text_splitter.split_documents(DataFrameLoader(bm25_df, page_content_column="page_content").load())

bm25_retriever = BM25Retriever.from_documents(bm25_documents)
print(f"BM25 retriever initialized with {len(bm25_documents)} documents.")

# 4. LLM for generation (Initialized with a default temperature, will be adjusted)
llm = ChatOpenAI(
    openai_api_key=OPENAI_API_KEY,
    model="gpt-4o-mini",
    temperature=0.7 # Default temperature, will be dynamically adjusted
)
print(f"LLM initialized: {llm.model_name} with default temperature={llm.temperature}")


# 5. Reranker Model
reranker_model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
reranker = CrossEncoder(reranker_model_name)
print(f"Reranker initialized: {reranker_model_name}")

# 6. Define a custom retrieval function that incorporates Hybrid Search, Filtering, and Reranking
async def get_filtered_hybrid_and_reranked_documents(query: str, filters: Optional[Dict[str, Any]] = None, top_k_ensemble: int = 40, top_k_rerank: int = 5) -> List[Document]:
    pinecone_filter = {}
    if filters:
        if 'publication_year' in filters and filters['publication_year']:
            pinecone_filter['publication_year'] = {"$eq": str(filters['publication_year'])}
        if 'author' in filters and filters['author']:
            pinecone_filter['first_author'] = {"$eq": str(filters['author'])}

    pinecone_retriever = pinecone_vectorstore.as_retriever(
        search_kwargs={"k": top_k_ensemble // 2, "filter": pinecone_filter}
    )

    bm25_retriever.k = top_k_ensemble // 2

    ensemble_retriever = EnsembleRetriever(
        retrievers=[pinecone_retriever, bm25_retriever],
        weights=[0.5, 0.5]
    )

    retrieved_docs = await ensemble_retriever.ainvoke(query)

    print(f"Retrieved {len(retrieved_docs)} documents from hybrid search (before reranking).")

    if not retrieved_docs:
        print("No documents retrieved from hybrid search.")
        return []

    sentence_pairs = [[query, doc.page_content] for doc in retrieved_docs]
    rerank_scores = reranker.predict(sentence_pairs)

    scored_docs = []
    for i, score in enumerate(rerank_scores):
        scored_docs.append({"doc": retrieved_docs[i], "score": score})

    scored_docs.sort(key=lambda x: x["score"], reverse=True)

    reranked_docs = [item["doc"] for item in scored_docs[:top_k_rerank]]
    print(f"Selected {len(reranked_docs)} documents after reranking.")

    return reranked_docs


# 7. Define the prompt template for QA
system_template = """You are an AI assistant designed to provide helpful, accurate, and respectful information about academic papers.
**Your core principles are:**
- **Unbiased:** Be strictly neutral and impartial. Avoid expressing or perpetuating any biases related to religion, gender, ethnicity, nationality, politics, socioeconomic status, or any other demographic characteristic.
- **Respectful:** Ensure all responses are courteous, inclusive, and professional.
- **Privacy-conscious:** Do not solicit, store, or reveal any personally identifiable information (PII) about yourself or others.
- **Safe & Responsible:**
    - If a user asks for legal, medical, or financial advice, you MUST politely decline and recommend they consult a qualified professional (e.g., "I am an AI and cannot provide legal advice. Please consult with a legal professional for such matters.").
    - If a query involves sensitive or potentially harmful topics, respond cautiously and redirect to appropriate, safe resources if possible, without generating harmful content.
    - Avoid fabricating information. If you cannot find a direct answer within the provided context, provide a generalized relevant answer if possible, rather than stating you don't know, but maintain factual accuracy.

**Formatting Instructions:**
- For numbered lists, use standard numbering (e.g., "1. Item one", "2. Item two").
- Ensure each numbered item is on a NEW LINE.
- Do NOT use asterisks (**) or other Markdown for bolding within the numbered list itself.
- For general text, you can use standard Markdown (e.g., **bold**, *italic*).
    
Use the following pieces of retrieved context to answer the question.
Keep the answer concise and to the point.

After providing the answer, suggest 2-3 short, relevant follow-up questions. Format these questions as a JSON array on a new line, like this:
["Follow-up question 1?", "Follow-up question 2?", "Follow-up question 3?"]

Context:
{context}
"""

QA_CHAIN_PROMPT = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(system_template),
    HumanMessagePromptTemplate.from_template("{question}"),
])


# --- Feedback Analysis Functions ---
def get_feedback_collection():
    if db is None:
        raise RuntimeError("Firestore DB is not initialized. Cannot access collection.")
    return db.collection(f"artifacts/{app_id}/public/data/user_feedback")

async def get_overall_feedback_counts() -> Dict[str, int]:
    """Fetches total counts for 'like', 'dislike', and 'neutral' feedback."""
    if db is None:
        print("WARNING: Firestore DB not initialized, returning empty feedback counts.")
        return {"like": 0, "dislike": 0, "neutral": 0}

    try:
        # Use asyncio.to_thread to run synchronous Firestore get() in a separate thread
        feedback_docs = await asyncio.to_thread(get_feedback_collection().get)
        counts = {"like": 0, "dislike": 0, "neutral": 0}
        for doc in feedback_docs:
            data = doc.to_dict()
            feedback_type = data.get("feedback_type")
            if feedback_type in counts:
                counts[feedback_type] += 1
        print(f"Fetched feedback counts: {counts}")
        return counts
    except Exception as e:
        print(f"Error fetching feedback counts from Firestore: {e}")
        return {"like": 0, "dislike": 0, "neutral": 0}

def calculate_helpfulness_score(feedback_counts: Dict[str, int]) -> float:
    """Calculates a helpfulness score based on feedback counts."""
    likes = feedback_counts.get("like", 0)
    dislikes = feedback_counts.get("dislike", 0)
    neutral = feedback_counts.get("neutral", 0)

    total_feedback = likes + dislikes + neutral
    if total_feedback == 0:
        return 0.0 # Neutral score if no feedback

    # A simple score: (Likes - Dislikes) / Total Feedback
    # This formula weights likes positively, dislikes negatively, and neutral as zero impact on the score.
    score = (likes - dislikes) / total_feedback
    print(f"Calculated helpfulness score: {score:.2f} (Likes: {likes}, Dislikes: {dislikes}, Neutral: {neutral})")
    return score

def adjust_llm_temperature(helpfulness_score: float) -> float:
    """Adjusts LLM temperature based on the helpfulness score."""
    # Desired temperature range
    MIN_TEMP = 0.2 # More deterministic/factual
    MAX_TEMP = 0.9 # More creative/exploratory
    DEFAULT_TEMP = 0.7 # A baseline if no feedback or neutral score

    # Map score (-1 to 1) to temperature (MAX_TEMP to MIN_TEMP for helpful, MIN_TEMP to MAX_TEMP for unhelpful)
    # A score of 1 (most helpful) should lead to MIN_TEMP (more deterministic)
    # A score of -1 (least helpful) should lead to MAX_TEMP (more creative/exploratory)
    # A score of 0 (neutral) should lead to roughly (MIN_TEMP + MAX_TEMP) / 2
    # Formula: new_temp = MIN_TEMP + (1 - score) * (MAX_TEMP - MIN_TEMP) / 2
    # This formula ensures that a higher helpfulness score (closer to 1) results in a lower temperature (closer to MIN_TEMP).
    # A lower helpfulness score (closer to -1) results in a higher temperature (closer to MAX_TEMP).

    if helpfulness_score is None:
        return DEFAULT_TEMP

    # Calculate the new temperature based on the helpfulness score
    # The (1 - helpfulness_score) term scales the score from [0, 2]
    # Dividing by 2 normalizes it to [0, 1]
    # Multiplying by (MAX_TEMP - MIN_TEMP) scales it to the desired temperature range
    # Adding MIN_TEMP shifts it to the correct starting point
    new_temperature = MIN_TEMP + (1 - helpfulness_score) * (MAX_TEMP - MIN_TEMP) / 2

    # Clamp the temperature to ensure it stays within the defined range
    new_temperature = max(MIN_TEMP, min(MAX_TEMP, new_temperature))
    print(f"Adjusted LLM temperature to: {new_temperature:.2f} based on helpfulness score: {helpfulness_score:.2f}")
    return new_temperature

# --- FastAPI Application ---
app = FastAPI(
    title="LangChain QA API",
    description="A simple API for question answering over academic papers with voice input, RAG, Hybrid Search, Reranking, Metadata Filters, User Feedback, and Reduced LLM calls.",
    version="1.0.0",
)

origins = [
    "http://localhost:8000",
    "http://127.0.0.1:8000",
    "null", # This is often required when serving index.html directly from file system (file://)
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request body model for the /ask endpoint
class AskRequest(BaseModel):
    session_id: Optional[str] = Field(None, description="Unique identifier for the chat session.")
    query: str = Field(..., description="The user's question.")
    filters: Optional[Dict[str, Any]] = Field(None, description="Optional metadata filters (e.g., {'publication_year': '2019', 'author': 'Sanh'}).")

# Response body model for the /ask endpoint
class AskResponse(BaseModel):
    session_id: str
    answer: str
    source_documents: List[dict] = []
    follow_up_questions: Optional[List[str]] = Field(None, description="Suggested follow-up questions.") # New field

# New Pydantic model for feedback
class FeedbackRequest(BaseModel):
    session_id: str = Field(..., description="The ID of the session the feedback belongs to.")
    query: str = Field(..., description="The user's query for which the feedback is given.")
    answer: str = Field(..., description="The AI's answer for which the feedback is given.")
    feedback_type: str = Field(..., description="Type of feedback: 'like', 'neutral' or 'dislike'.")
    timestamp: str = Field(..., description="Timestamp of the feedback.")
    source_docs: Optional[List[dict]] = Field(None, description="Optional list of source documents used for the answer.")


@app.post("/submit_feedback")
async def submit_feedback(request: FeedbackRequest):
    """
    Submits user feedback (like/dislike) to Firestore.
    """
    print(f"DEBUG: Received feedback request for feedback: {request.dict()}")
    if db is None:
        print("ERROR: Firestore DB is not initialized. Cannot submit feedback.")
        raise HTTPException(status_code=500, detail="Firestore database not available for feedback.")

    try:
        feedback_data = request.dict()
        feedback_data["server_timestamp"] = firestore.SERVER_TIMESTAMP
        doc_id = f"{request.session_id}_{datetime.datetime.now().isoformat().replace('.', '-')}"
        # Use asyncio.to_thread to run synchronous Firestore set() in a separate thread
        await asyncio.to_thread(get_feedback_collection().document(doc_id).set, feedback_data)
        print(f"Feedback submitted for session {request.session_id}, type: {request.feedback_type}")
        return {"message": "Feedback submitted successfully!"}
    except Exception as e:
        print(f"Error submitting feedback: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to submit feedback: {e}")


@app.post("/ask", response_model=AskResponse)
async def ask_question(request: AskRequest):
    """
    Answers a question based on the indexed documents, applying filters,
    and reranking retrieved documents. Groundedness check is disabled.
    Dynamically adjusts LLM temperature based on overall feedback.
    """
    try:
        session_id = request.session_id if request.session_id else str(uuid.uuid4())
        print(f"Processing query for session ID: {session_id}")
        print(f"DEBUG: Incoming AskRequest payload: {json.dumps(request.dict(), indent=2)}")

        # --- NEW: Fetch feedback and adjust LLM temperature ---
        overall_feedback_counts = await get_overall_feedback_counts()
        helpfulness_score = calculate_helpfulness_score(overall_feedback_counts)
        adjusted_temperature = adjust_llm_temperature(helpfulness_score)

        # Update the LLM's temperature for this request
        llm.temperature = adjusted_temperature
        print(f"Using LLM with dynamically adjusted temperature: {llm.temperature}")

        # Re-initialize the qa_chain with the updated LLM temperature
        # This is important because RetrievalQA chain captures the LLM at its creation
        qa_chain_dynamic = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=pinecone_vectorstore.as_retriever(search_kwargs={"k": 1}), # Placeholder, overridden by custom_retriever
            return_source_documents=True,
            chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
        )
        print("RetrievalQA chain dynamically re-initialized with new temperature.")
        # --- END NEW ---


        current_filters = request.filters if request.filters is not None else {}
        print(f"DEBUG: Filters being used: {current_filters}")

        retrieved_documents = await get_filtered_hybrid_and_reranked_documents(
            query=request.query,
            filters=current_filters,
            top_k_ensemble=40,
            top_k_rerank=5
        )

        if not retrieved_documents:
            print("DEBUG: No documents retrieved from hybrid search for the query.")
            answer = "I couldn't find very specific information, but here's a generalized response based on common knowledge."
            # Still provide an opportunity for follow-up questions related to the generalized answer
            follow_up_questions = ["What are the key concepts in AI?", "How can I find more specific papers?", "What is RAG?"]
            return AskResponse(session_id=session_id, answer=answer, follow_up_questions=follow_up_questions)

        class CustomDocsRetriever:
            def __init__(self, docs: List[Document]):
                self.docs = docs
            def get_relevant_documents(self, query: str) -> List[Document]:
                return self.docs
            async def aget_relevant_documents(self, query: str) -> List[Document]:
                return self.docs

        custom_retriever = CustomDocsRetriever(retrieved_documents)

        try:
            # Use the dynamically created qa_chain_dynamic
            result = await qa_chain_dynamic.ainvoke({
                "query": request.query,
                "context": custom_retriever.get_relevant_documents(request.query)
            })

            generated_raw_response = result["result"]
            generated_answer = generated_raw_response
            follow_up_questions = []

            # Attempt to parse follow-up questions from the end of the generated answer
            try:
                start_json = generated_raw_response.rfind('[')
                end_json = generated_raw_response.rfind(']')
                if start_json != -1 and end_json != -1 and end_json > start_json:
                    json_str = generated_raw_response[start_json : end_json + 1]
                    potential_questions = json.loads(json_str)
                    if isinstance(potential_questions, list) and all(isinstance(q, str) for q in potential_questions):
                        follow_up_questions = potential_questions
                        generated_answer = generated_raw_response[:start_json].strip()
                        print(f"DEBUG: Parsed follow-up questions: {follow_up_questions}")
                    else:
                        print("DEBUG: JSON array found but content not a list of strings. Keeping as part of answer.")
                else:
                    print("DEBUG: No JSON array pattern for follow-up questions found.")
            except json.JSONDecodeError as e:
                print(f"DEBUG: Could not decode JSON for follow-up questions: {e}. Keeping as part of answer.")
            except Exception as e:
                print(f"DEBUG: Unexpected error parsing follow-up questions: {e}. Keeping as part of answer.")


            source_docs_info = []
            if "source_documents" in result and result["source_documents"]:
                for doc in result["source_documents"]:
                    meta_to_display = {k: v for k, v in doc.metadata.items() if k not in ['text_chunk_id', 'vector_id']}
                    source_docs_info.append({
                        "page_content": doc.page_content,
                        "metadata": meta_to_display
                    })

            return AskResponse(session_id=session_id, answer=generated_answer, source_documents=source_docs_info, follow_up_questions=follow_up_questions)

        except Exception as chain_e:
            print(f"ERROR: During QA chain invocation: {chain_e}")
            raise HTTPException(status_code=500, detail=f"Internal processing error during QA chain: {chain_e}")

    except Exception as e:
        print(f"ERROR: Unhandled error in /ask endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred in /ask endpoint: {e}")

# Basic root endpoint for health check
@app.get("/")
async def read_root():
    return {"message": "LangChain QA API is running. Use /ask to query or /submit_feedback to send feedback."}
