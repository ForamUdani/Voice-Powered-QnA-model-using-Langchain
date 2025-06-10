# main.py

import os
from typing import List, Tuple, Dict, Any, Optional
from dotenv import load_dotenv
import uuid # For generating session IDs
import datetime # For timestamps

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain.chains import ConversationalRetrievalChain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

# Firebase Admin SDK imports
import firebase_admin
from firebase_admin import credentials, firestore

# Load environment variables
load_dotenv()

# --- Configuration ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
PINECONE_INDEX_NAME = "ai-arxiv-chunks-langchain" # Must match the index name used in ingest.py

if not all([OPENAI_API_KEY, PINECONE_API_KEY, PINECONE_ENVIRONMENT]):
    raise ValueError("Please set OPENAI_API_KEY, PINECONE_API_KEY, and PINECONE_ENVIRONMENT environment variables.")

# --- Firebase Initialization ---
db = None
app_id = "default-app-id" # Default for local testing, will be overridden in Canvas

try:
    # Check if __firebase_config and __app_id are defined (for Canvas environment)
    if '__firebase_config' in globals() and '__app_id' in globals():
        # IMPORTANT: In Canvas, __firebase_config is a stringified JSON.
        # You need to parse it if it's not already handled by the environment.
        # However, for Python in Canvas, it might be directly available as a dict.
        # Let's assume it's a dict for simplicity, if it fails, you might need json.loads(__firebase_config)
        firebase_config = globals().get('__firebase_config')
        app_id = globals().get('__app_id')

        # Use credentials.Certificate for Service Account key based auth
        # If __firebase_config is a dict, use it directly.
        # If it's a JSON string, uncomment: cred = credentials.Certificate(json.loads(firebase_config))
        cred = credentials.Certificate(firebase_config)

        if not firebase_admin._apps: # Prevent re-initialization
            firebase_admin.initialize_app(cred)
        db = firestore.client()
        print(f"Firebase initialized for Canvas app ID: {app_id}")
    else:
        # Fallback for local development if not in Canvas environment
        # Replace 'path/to/your/serviceAccountKey.json' with your actual path
        # Ensure you've downloaded your Firebase service account key JSON file
        # from Project settings > Service accounts > Generate new private key
        cred = credentials.Certificate("serviceAccountKey.json") # Ensure this file exists for local dev
        if not firebase_admin._apps:
            firebase_admin.initialize_app(cred)
        db = firestore.client()
        print("Firebase initialized using serviceAccountKey.json for local development.")
except Exception as e:
    print(f"Error initializing Firebase: {e}")
    db = None # Ensure db is None if initialization fails

if db is None:
    # If Firebase init fails, we cannot proceed with persistence features.
    # For a robust app, you might allow non-persistent features,
    # but for feedback, it's mandatory.
    raise RuntimeError("Firestore database not initialized. Cannot proceed with feedback persistence.")


# --- LangChain Setup ---
print("Initializing LangChain components...")

# 1. Embeddings
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# 2. Vector Store (Pinecone)
vectorstore = PineconeVectorStore(
    index_name=PINECONE_INDEX_NAME,
    embedding=embeddings
)
print(f"Pinecone vectorstore initialized for index: {PINECONE_INDEX_NAME}")

# 3. LLM for generation
llm = ChatOpenAI(
    openai_api_key=OPENAI_API_KEY,
    model="gpt-4o-mini",
    temperature=0.7
)
print(f"LLM initialized: {llm.model_name}")

# 4. Reranker Model
reranker_model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
reranker = CrossEncoder(reranker_model_name)
print(f"Reranker initialized: {reranker_model_name}")

# 5. Define a custom retrieval function that incorporates filtering and reranking
def get_filtered_and_reranked_documents(query: str, filters: Optional[Dict[str, Any]] = None, top_k_retrieval: int = 20, top_k_rerank: int = 5) -> List[Document]:
    """
    Retrieves documents from Pinecone, applies filters, and then reranks them.
    """
    pinecone_filter = {}
    if filters:
        if 'publication_year' in filters and filters['publication_year']:
            pinecone_filter['publication_year'] = {"$eq": str(filters['publication_year'])}
        if 'author' in filters and filters['author']:
            pinecone_filter['first_author'] = {"$eq": str(filters['author'])}

    print(f"Pinecone filter being applied: {pinecone_filter}")

    retrieved_docs = vectorstore.as_retriever(
        search_kwargs={"k": top_k_retrieval, "filter": pinecone_filter}
    ).invoke(query)

    print(f"Retrieved {len(retrieved_docs)} documents from Pinecone.")

    if not retrieved_docs:
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

# 6. Define the prompt template for conversational retrieval
system_template = """You are an AI assistant for question-answering over academic papers.
Use the following pieces of retrieved context to answer the question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Keep the answer concise and to the point.

Context:
{context}
"""

qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vectorstore.as_retriever(search_kwargs={"k": 1}), # Dummy retriever, will be overridden
    condense_question_prompt=ChatPromptTemplate.from_messages(
        [
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
            ("ai", "Given the above conversation and a follow up question, rephrase the follow up question to be a standalone question."),
        ]
    ),
    combine_docs_chain_kwargs={"prompt": ChatPromptTemplate.from_messages([
        ("system", system_template),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ])},
    return_source_documents=True
)
print("ConversationalRetrievalChain initialized.")

# --- FastAPI Application ---
app = FastAPI(
    title="LangChain QA API",
    description="A simple API for question answering over academic papers with voice input, RAG, Reranking, Metadata Filters, and User Feedback.",
    version="1.0.0",
)

# Request body model for the /ask endpoint
class AskRequest(BaseModel):
    session_id: Optional[str] = Field(None, description="Unique identifier for the chat session.")
    query: str = Field(..., description="The user's question.")
    chat_history: List[Tuple[str, str]] = Field([], description="List of (human_message, ai_message) tuples for conversation history.")
    filters: Optional[Dict[str, Any]] = Field(None, description="Optional metadata filters (e.g., {'publication_year': '2019', 'author': 'Sanh'}).")

# Response body model for the /ask endpoint
class AskResponse(BaseModel):
    session_id: str # Return the session ID
    answer: str
    source_documents: List[dict] = [] # List of dictionaries for source info

# New Pydantic model for feedback
class FeedbackRequest(BaseModel):
    session_id: str = Field(..., description="The ID of the session the feedback belongs to.")
    query: str = Field(..., description="The user's query for which the feedback is given.")
    answer: str = Field(..., description="The AI's answer for which the feedback is given.")
    feedback_type: str = Field(..., description="Type of feedback: 'like' or 'dislike'.")
    timestamp: str = Field(..., description="Timestamp of the feedback.")
    source_docs: Optional[List[dict]] = Field(None, description="Optional list of source documents used for the answer.")


# Define the Firestore collection path for feedback
def get_feedback_collection():
    # Store in public data so it can be analyzed
    return db.collection(f"artifacts/{app_id}/public/data/user_feedback")

@app.post("/submit_feedback")
async def submit_feedback(request: FeedbackRequest):
    """
    Submits user feedback (like/dislike) to Firestore.
    """
    try:
        feedback_data = request.dict()
        # Add a server-side timestamp for accuracy
        feedback_data["server_timestamp"] = firestore.SERVER_TIMESTAMP

        # Use a unique ID for each feedback entry, could be a combination of session_id and timestamp
        doc_id = f"{request.session_id}_{datetime.datetime.now().isoformat()}"
        get_feedback_collection().document(doc_id).set(feedback_data)
        print(f"Feedback submitted for session {request.session_id}, type: {request.feedback_type}")
        return {"message": "Feedback submitted successfully!"}
    except Exception as e:
        print(f"Error submitting feedback: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to submit feedback: {e}")


@app.post("/ask", response_model=AskResponse)
async def ask_question(request: AskRequest):
    """
    Answers a question based on the indexed documents, applying filters,
    and reranking retrieved documents.
    """
    try:
        # Generate session ID if not provided (first time user)
        session_id = request.session_id if request.session_id else str(uuid.uuid4())
        print(f"Processing query for session ID: {session_id}")

        # The chat_history from the frontend is already the current state.
        # We're no longer loading full history from Firestore on every 'ask'
        # if the user hasn't explicitly enabled full history persistence.
        # This current implementation assumes `request.chat_history` is what the
        # frontend currently has and sends.
        formatted_chat_history_for_llm = []
        for human_msg, ai_msg in request.chat_history:
            formatted_chat_history_for_llm.append(HumanMessage(content=human_msg))
            formatted_chat_history_for_llm.append(AIMessage(content=ai_msg))

        # Step 1: Retrieve and Rerank documents using our custom function
        retrieved_documents = get_filtered_and_reranked_documents(
            query=request.query,
            filters=request.filters,
            top_k_retrieval=20,
            top_k_rerank=5
        )

        if not retrieved_documents:
            answer = "I couldn't find any relevant information for your query, even with the applied filters."
            return AskResponse(session_id=session_id, answer=answer)

        class CustomDocsRetriever:
            def __init__(self, docs: List[Document]):
                self.docs = docs
            def get_relevant_documents(self, query: str) -> List[Document]:
                return self.docs
            async def aget_relevant_documents(self, query: str) -> List[Document]:
                return self.docs

        custom_retriever = CustomDocsRetriever(retrieved_documents)

        # Invoke the QA chain with the custom retriever
        result = await qa_chain.ainvoke({
            "question": request.query,
            "chat_history": formatted_chat_history_for_llm,
        }, config={"retriever": custom_retriever})

        answer = result["answer"]
        source_docs_info = []
        if "source_documents" in result and result["source_documents"]:
            for doc in result["source_documents"]:
                meta_to_display = {k: v for k, v in doc.metadata.items() if k not in ['text_chunk_id', 'vector_id']}
                source_docs_info.append({
                    "page_content": doc.page_content,
                    "metadata": meta_to_display
                })

        return AskResponse(session_id=session_id, answer=answer, source_documents=source_docs_info)

# Basic root endpoint for health check
@app.get("/")
async def read_root():
    return {"message": "LangChain QA API is running. Use /ask to query or /submit_feedback to send feedback."}

