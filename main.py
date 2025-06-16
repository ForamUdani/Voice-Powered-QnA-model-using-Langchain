# main.py

import os
from typing import List, Tuple, Dict, Any, Optional
from dotenv import load_dotenv # Ensure this is at the very top
import uuid
import datetime
import json

# Load environment variables including LangSmith keys
load_dotenv()

# --- Configuration ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
PINECONE_INDEX_NAME = "ai-arxiv-chunks-langchain"

# LangSmith specific environment variables (set these in your .env file)
# os.environ["LANGCHAIN_TRACING_V2"] = "true" # Already loaded by dotenv
# os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY") # Already loaded by dotenv
# os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", "Voice_QA_Bot") # Default project name if not set

if not all([OPENAI_API_KEY, PINECONE_API_KEY, PINECONE_ENVIRONMENT]):
    raise ValueError("Please set OPENAI_API_KEY, PINECONE_API_KEY, and PINECONE_ENVIRONMENT environment variables.")

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain.chains import ConversationalRetrievalChain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

# New imports for Hybrid Search
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_community.document_loaders import DataFrameLoader
from datasets import load_dataset
import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Firebase Admin SDK imports
import firebase_admin
from firebase_admin import credentials, firestore


# --- Firebase Initialization ---
db = None
app_id = "default-app-id" # Default for local testing, will be overridden in Canvas

try:
    if '__firebase_config' in globals() and '__app_id' in globals():
        firebase_config_dict = json.loads(globals().get('__firebase_config'))
        app_id = globals().get('__app_id')

        cred = credentials.Certificate(firebase_config_dict)
        if not firebase_admin._apps:
            firebase_admin.initialize_app(cred)
        db = firestore.client()
        print(f"Firebase initialized for Canvas app ID: {app_id}")
    else:
        cred = credentials.Certificate("serviceAccountKey.json")
        if not firebase_admin._apps:
            firebase_admin.initialize_app(cred)
        db = firestore.client()
        print("Firebase initialized using serviceAccountKey.json for local development.")
except Exception as e:
    print(f"Error initializing Firebase: {e}")
    db = None

if db is None:
    raise RuntimeError("Firestore database not initialized. Cannot proceed with feedback persistence.")


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
bm25_dataset = load_dataset("jamescalam/ai-arxiv-chunked", split="train")
bm25_df = bm25_dataset.to_pandas()
bm25_df = bm25_df.dropna(subset=["chunk"])
bm25_df["page_content"] = bm25_df.apply(lambda row: f"Title: {row['title']}\n\nChunk: {row['chunk']}", axis=1)

bm25_text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
    is_separator_regex=False,
)
bm25_documents = bm25_text_splitter.split_documents(DataFrameLoader(bm25_df, page_content_column="page_content").load())

bm25_retriever = BM25Retriever.from_documents(bm25_documents)
print(f"BM25 retriever initialized with {len(bm25_documents)} documents.")

# 4. LLM for generation
llm = ChatOpenAI(
    openai_api_key=OPENAI_API_KEY,
    model="gpt-4o-mini",
    temperature=0.2
)
print(f"LLM initialized: {llm.model_name} with temperature={llm.temperature}")

# LLM for Groundedness Check
groundedness_llm = ChatOpenAI(
    openai_api_key=OPENAI_API_KEY,
    model="gpt-3.5-turbo",
    temperature=0.0
)
print(f"Groundedness LLM initialized: {groundedness_llm.model_name} with temperature={groundedness_llm.temperature}")


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

# 7. Groundedness Check Function (Anti-Hallucination)
async def check_groundedness(answer: str, retrieved_docs: List[Document]) -> bool:
    if not retrieved_docs:
        return False

    context_str = "\n".join([doc.page_content for doc in retrieved_docs])

    groundedness_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an AI assistant that evaluates if an 'Answer' is fully supported by the provided 'Context Snippets'.
        Respond with 'YES' if the answer is completely verifiable by the snippets.
        Respond with 'NO' if any part of the answer is not present, contradicts, or cannot be inferred from the snippets.
        Do not elaborate, just 'YES' or 'NO'."""),
        ("user", f"Context Snippets:\n```\n{context_str}\n```\n\nAnswer:\n```\n{answer}\n```\n\nIs the Answer fully supported by the Context Snippets? (YES/NO)")
    ])

    try:
        response = await groundedness_llm.ainvoke(groundedness_prompt.format_messages(answer=answer, context_str=context_str))
        grounded_check = response.content.strip().upper()
        print(f"Groundedness check result: {grounded_check}")
        return grounded_check == "YES"
    except Exception as e:
        print(f"Error during groundedness check: {e}")
        return True # Default to True if check fails to avoid blocking, but log the error

# 8. Define the prompt template for conversational retrieval
system_template = """You are an AI assistant for question-answering over academic papers.
**Strictly adhere to the provided context.**
Use the following pieces of retrieved context to answer the question.
If the context does not contain enough information to answer the question, or if you are unsure, **you must state "I cannot find a definitive answer to this question based on the provided information."**
Do not invent facts or extrapolate beyond the given context. Keep the answer concise and to the point.

Context:
{context}
"""

qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=pinecone_vectorstore.as_retriever(search_kwargs={"k": 1}), # Dummy retriever, will be overridden by custom invocation
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
    description="A simple API for question answering over academic papers with voice input, RAG, Hybrid Search, Reranking, Metadata Filters, User Feedback, and Hallucination Control.",
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
    session_id: str
    answer: str
    source_documents: List[dict] = []

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
    return db.collection(f"artifacts/{app_id}/public/data/user_feedback")

@app.post("/submit_feedback")
async def submit_feedback(request: FeedbackRequest):
    """
    Submits user feedback (like/dislike) to Firestore.
    """
    try:
        feedback_data = request.dict()
        feedback_data["server_timestamp"] = firestore.SERVER_TIMESTAMP
        doc_id = f"{request.session_id}_{datetime.datetime.now().isoformat().replace('.', '-')}"
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
    and reranking retrieved documents. Includes hallucination control.
    """
    try:
        session_id = request.session_id if request.session_id else str(uuid.uuid4())
        print(f"Processing query for session ID: {session_id}")

        formatted_chat_history_for_llm = []
        for human_msg, ai_msg in request.chat_history:
            formatted_chat_history_for_llm.append(HumanMessage(content=human_msg))
            formatted_chat_history_for_llm.append(AIMessage(content=ai_msg))

        retrieved_documents = await get_filtered_hybrid_and_reranked_documents(
            query=request.query,
            filters=request.filters,
            top_k_ensemble=40,
            top_k_rerank=5
        )

        if not retrieved_documents:
            answer = "I cannot find a definitive answer to this question based on the provided information."
            return AskResponse(session_id=session_id, answer=answer)

        class CustomDocsRetriever:
            def __init__(self, docs: List[Document]):
                self.docs = docs
            def get_relevant_documents(self, query: str) -> List[Document]:
                return self.docs
            async def aget_relevant_documents(self, query: str) -> List[Document]:
                return self.docs

        custom_retriever = CustomDocsRetriever(retrieved_documents)

        # Corrected: Added a try-except block around qa_chain.ainvoke
        try: # This is the corrected try block
            result = await qa_chain.ainvoke({
                "question": request.query,
                "chat_history": formatted_chat_history_for_llm,
            }, config={"retriever": custom_retriever})

            generated_answer = result["answer"]
            source_docs_info = []

            if "source_documents" in result and result["source_documents"]:
                for doc in result["source_documents"]:
                    meta_to_display = {k: v for k, v in doc.metadata.items() if k not in ['text_chunk_id', 'vector_id']}
                    source_docs_info.append({
                        "page_content": doc.page_content,
                        "metadata": meta_to_display
                    })

            # Perform Groundedness Check
            is_grounded = await check_groundedness(generated_answer, retrieved_documents)

            if not is_grounded:
                answer = "I cannot find a definitive answer to this question based on the provided information, as I could not verify all parts of the generated response from the retrieved context. Please try rephrasing."
                source_docs_info = []
            else:
                answer = generated_answer

            return AskResponse(session_id=session_id, answer=answer, source_documents=source_docs_info)

        except Exception as chain_e: # Added an except block here
            print(f"Error during QA chain invocation: {chain_e}")
            raise HTTPException(status_code=500, detail=f"Error generating answer: {chain_e}")

    except Exception as e:
        print(f"Unhandled error in /ask endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")

# Basic root endpoint for health check
@app.get("/")
async def read_root():
    return {"message": "LangChain QA API is running. Use /ask to query or /submit_feedback to send feedback."}

