# main.py

import os
from typing import List, Tuple, Dict, Any, Optional
from dotenv import load_dotenv
import uuid # For generating session IDs

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
        firebase_config = JSON.parse(__firebase_config)
        app_id = __app_id
        # Initialize Firebase with the provided config
        if not firebase_admin._apps: # Prevent re-initialization
            firebase_admin.initialize_app(credentials.Certificate(firebase_config))
        db = firestore.client()
        print(f"Firebase initialized for Canvas app ID: {app_id}")
    else:
        # Fallback for local development if not in Canvas environment
        # Replace 'path/to/your/serviceAccountKey.json' with your actual path
        # Ensure you've downloaded your Firebase service account key JSON file
        cred = credentials.Certificate("serviceAccountKey.json")
        if not firebase_admin._apps:
            firebase_admin.initialize_app(cred)
        db = firestore.client()
        print("Firebase initialized using serviceAccountKey.json for local development.")
except Exception as e:
    print(f"Error initializing Firebase: {e}")
    db = None # Ensure db is None if initialization fails

if db is None:
    raise RuntimeError("Firestore database not initialized. Cannot proceed with persistence.")


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
    description="A simple API for question answering over academic papers with voice input, RAG, Reranking, Metadata Filters, and Firestore Persistence.",
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

class HistoryResponse(BaseModel):
    session_id: str
    chat_history: List[Tuple[str, str]]

# Define the Firestore collection path for chat sessions
def get_chat_sessions_collection():
    return db.collection(f"artifacts/{app_id}/public/data/chat_sessions")

@app.get("/get_history", response_model=HistoryResponse)
async def get_history(session_id: str = Query(..., description="The session ID to retrieve chat history for.")):
    """
    Retrieves the chat history for a given session ID from Firestore.
    """
    try:
        doc_ref = get_chat_sessions_collection().document(session_id)
        doc = doc_ref.get()

        if doc.exists:
            data = doc.to_dict()
            retrieved_history = data.get("chat_history", [])
            print(f"Retrieved history for session {session_id}: {len(retrieved_history)} turns.")
            return HistoryResponse(session_id=session_id, chat_history=retrieved_history)
        else:
            print(f"No history found for session {session_id}. Returning empty.")
            return HistoryResponse(session_id=session_id, chat_history=[])
    except Exception as e:
        print(f"Error fetching history for session {session_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Error fetching history: {e}")


@app.post("/ask", response_model=AskResponse)
async def ask_question(request: AskRequest):
    """
    Answers a question based on the indexed documents, maintaining conversation history,
    applying filters, and reranking retrieved documents. History is persisted in Firestore.
    """
    try:
        session_id = request.session_id if request.session_id else str(uuid.uuid4())
        print(f"Processing query for session ID: {session_id}")

        # Load existing chat history from Firestore if available
        existing_chat_history_doc = get_chat_sessions_collection().document(session_id).get()
        current_chat_history = []
        if existing_chat_history_doc.exists:
            current_chat_history = existing_chat_history_doc.to_dict().get("chat_history", [])
            print(f"Loaded {len(current_chat_history)} turns from existing history.")

        # Add the new human query to the history
        # Note: request.chat_history from frontend is often just the *current* state.
        # We need to ensure we use the full history from Firestore + current query for LLM.
        # LangChain's ConversationalRetrievalChain takes `chat_history` for the LLM.
        # We'll pass `current_chat_history` (from Firestore) to the chain for rephrasing.
        formatted_chat_history_for_llm = []
        for human_msg, ai_msg in current_chat_history:
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
            # Store the current turn (human query, no answer)
            current_chat_history.append((request.query, answer))
            get_chat_sessions_collection().document(session_id).set({"chat_history": current_chat_history})
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

        # --- Persist updated chat history to Firestore ---
        current_chat_history.append((request.query, answer))
        get_chat_sessions_collection().document(session_id).set({"chat_history": current_chat_history})
        print(f"Saved {len(current_chat_history)} turns for session {session_id}.")

        return AskResponse(session_id=session_id, answer=answer, source_documents=source_docs_info)

    except Exception as e:
        print(f"Error processing request: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Basic root endpoint for health check
@app.get("/")
async def read_root():
    return {"message": "LangChain QA API is running. Use /ask to query or /get_history to retrieve session history."}

