# main.py

import os
from typing import List, Tuple, Dict, Any, Optional
from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain.chains import ConversationalRetrievalChain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.documents import Document # Import Document class
from sentence_transformers import CrossEncoder # For reranking

# Load environment variables
load_dotenv()

# --- Configuration ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
PINECONE_INDEX_NAME = "ai-arxiv-chunks-langchain" # Must match the index name used in ingest.py

if not all([OPENAI_API_KEY, PINECONE_API_KEY, PINECONE_ENVIRONMENT]):
    raise ValueError("Please set OPENAI_API_KEY, PINECONE_API_KEY, and PINECONE_ENVIRONMENT environment variables.")

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
# Using a good general-purpose cross-encoder for reranking.
# Make sure to install sentence-transformers: pip install sentence-transformers
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
        # Example: Convert UI filters to Pinecone filter format
        if 'publication_year' in filters and filters['publication_year']:
            # Assuming year is passed as a string like "2019"
            pinecone_filter['publication_year'] = {"$eq": str(filters['publication_year'])}
        if 'author' in filters and filters['author']:
            # Assuming author is passed as a string
            pinecone_filter['first_author'] = {"$eq": str(filters['author'])}
        # Add more filter conditions as needed based on your metadata

    print(f"Pinecone filter being applied: {pinecone_filter}")

    # Step 1: Retrieve initial documents from Pinecone with filtering
    # We retrieve more documents than we need for the final answer to allow reranking to work effectively.
    retrieved_docs = vectorstore.as_retriever(
        search_kwargs={"k": top_k_retrieval, "filter": pinecone_filter}
    ).invoke(query)

    print(f"Retrieved {len(retrieved_docs)} documents from Pinecone.")

    if not retrieved_docs:
        return []

    # Step 2: Rerank the retrieved documents
    # The reranker takes (query, document.page_content) pairs
    sentence_pairs = [[query, doc.page_content] for doc in retrieved_docs]
    rerank_scores = reranker.predict(sentence_pairs)

    # Combine documents with their rerank scores
    scored_docs = []
    for i, score in enumerate(rerank_scores):
        scored_docs.append({"doc": retrieved_docs[i], "score": score})

    # Sort documents by score in descending order
    scored_docs.sort(key=lambda x: x["score"], reverse=True)

    # Select the top_k_rerank documents
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

# 7. ConversationalRetrievalChain setup
# We'll use a custom chain for retrieval to integrate reranking and filtering directly.
# The 'retriever' here will be a dummy one as we handle retrieval manually before calling the chain.
# Alternatively, you could wrap get_filtered_and_reranked_documents in a custom LangChain Retriever
# but for clarity, we'll do it manually before invoking the chain.
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vectorstore.as_retriever(search_kwargs={"k": 1}), # Dummy retriever, will be overridden by custom invocation
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
    description="A simple API for question answering over academic papers using LangChain, Pinecone, Reranking, and Metadata Filters.",
    version="1.0.0",
)

# Request body model for the /ask endpoint
class AskRequest(BaseModel):
    query: str = Field(..., description="The user's question.")
    chat_history: List[Tuple[str, str]] = Field([], description="List of (human_message, ai_message) tuples for conversation history.")
    filters: Optional[Dict[str, Any]] = Field(None, description="Optional metadata filters (e.g., {'publication_year': '2019', 'author': 'Sanh'}).")

# Response body model for the /ask endpoint
class AskResponse(BaseModel):
    answer: str
    source_documents: List[dict] = [] # List of dictionaries for source info

@app.post("/ask", response_model=AskResponse)
async def ask_question(request: AskRequest):
    """
    Answers a question based on the indexed documents, maintaining conversation history,
    applying filters, and reranking retrieved documents.
    """
    try:
        # Convert chat history to LangChain message format
        formatted_chat_history = []
        for human_msg, ai_msg in request.chat_history:
            formatted_chat_history.append(HumanMessage(content=human_msg))
            formatted_chat_history.append(AIMessage(content=ai_msg))

        # Step 1: Retrieve and Rerank documents using our custom function
        retrieved_documents = get_filtered_and_reranked_documents(
            query=request.query,
            filters=request.filters,
            top_k_retrieval=20, # Retrieve more initially
            top_k_rerank=5      # Select top 5 after reranking
        )

        if not retrieved_documents:
            # If no documents are found, respond that no relevant info was found.
            answer = "I couldn't find any relevant information for your query, even with the applied filters."
            return AskResponse(answer=answer)

        # Step 2: Pass the reranked documents directly to the QA chain
        # We need to manually construct the input for the combine_docs_chain part
        # because ConversationalRetrievalChain doesn't directly take 'documents' in invoke.
        # Instead, we are using its internal mechanism where its retriever would fetch.
        # To pass our pre-fetched documents, we can adapt the chain's invocation.
        # A more direct way is to use a custom chain or ensure the documents are passed correctly.

        # For simplicity with ConversationalRetrievalChain, we will directly
        # pass the documents via a custom retriever in the invoke call,
        # or more correctly, adapt the chain itself if needed.
        # A simpler way to inject documents: Langchain's ConversationalRetrievalChain.from_llm
        # takes a `retriever` argument. We can create a custom retriever that always
        # returns our `retrieved_documents`.

        # Create a simple custom retriever that returns the pre-fetched documents
        # This is a common pattern when you handle retrieval outside the standard chain.
        class CustomDocsRetriever:
            def __init__(self, docs: List[Document]):
                self.docs = docs
            def get_relevant_documents(self, query: str) -> List[Document]:
                return self.docs
            async def aget_relevant_documents(self, query: str) -> List[Document]:
                return self.docs

        custom_retriever = CustomDocsRetriever(retrieved_documents)

        # Invoke the QA chain with the custom retriever for this specific call
        # The chain will use this custom retriever to get context
        result = await qa_chain.ainvoke({ # Use ainvoke for async operations
            "question": request.query,
            "chat_history": formatted_chat_history,
            "context": retrieved_documents # This might be redundat if custom_retriever is used
                                          # and the chain is designed to pass context directly.
                                          # `ConversationalRetrievalChain` primarily uses its internal retriever.
        }, config={"retriever": custom_retriever}) # Pass custom retriever in config

        answer = result["answer"]
        source_docs_info = []
        if "source_documents" in result and result["source_documents"]:
            for doc in result["source_documents"]:
                # Exclude internal metadata not relevant to the user
                meta_to_display = {k: v for k, v in doc.metadata.items() if k not in ['text_chunk_id', 'vector_id']}
                source_docs_info.append({
                    "page_content": doc.page_content,
                    "metadata": meta_to_display
                })

        return AskResponse(answer=answer, source_documents=source_docs_info)

    except Exception as e:
        print(f"Error processing request: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Basic root endpoint for health check
@app.get("/")
async def read_root():
    return {"message": "LangChain QA API is running. Use /ask to query."}

