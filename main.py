# main.py

import os
from typing import List, Tuple
from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain.chains import ConversationalRetrievalChain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

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
# We don't need to initialize Pinecone client explicitly here,
# PineconeVectorStore will handle it internally using the environment variables.
vectorstore = PineconeVectorStore(
    index_name=PINECONE_INDEX_NAME,
    embedding=embeddings
)
retriever = vectorstore.as_retriever()
print(f"Pinecone retriever initialized for index: {PINECONE_INDEX_NAME}")

# 3. LLM for generation
llm = ChatOpenAI(
    openai_api_key=OPENAI_API_KEY,
    model="gpt-4o-mini",  # Using gpt-4o-mini as per your LightRag code
    temperature=0.7      # Adjust temperature for creativity vs. factualness
)
print(f"LLM initialized: {llm.model_name}")

# 4. Define the prompt template for conversational retrieval
# This prompt is crucial for guiding the LLM to use retrieved context and maintain conversation history.
system_template = """You are an AI assistant for question-answering over academic papers.
Use the following pieces of retrieved context to answer the question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Keep the answer concise and to the point.

Context:
{context}
"""

# The `ConversationalRetrievalChain` handles combining the question and chat history
# to form a standalone question for the retriever, and then generates the final answer.
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    # This part specifies how the question and chat history are combined into a standalone question
    # for the retriever.
    # We'll use a simple prompt for this.
    condense_question_prompt=ChatPromptTemplate.from_messages(
        [
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
            ("ai", "Given the above conversation and a follow up question, rephrase the follow up question to be a standalone question."),
        ]
    ),
    # This part specifies the prompt used for the final answer generation
    combine_docs_chain_kwargs={"prompt": ChatPromptTemplate.from_messages([
        ("system", system_template),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ])},
    return_source_documents=True # Optionally return the documents that were used
)
print("ConversationalRetrievalChain initialized.")

# --- FastAPI Application ---
app = FastAPI(
    title="LangChain QA API",
    description="A simple API for question answering over academic papers using LangChain and Pinecone.",
    version="1.0.0",
)

# Request body model for the /ask endpoint
class AskRequest(BaseModel):
    query: str
    chat_history: List[Tuple[str, str]] = [] # List of (human_message, ai_message) tuples

# Response body model for the /ask endpoint
class AskResponse(BaseModel):
    answer: str
    source_documents: List[dict] = [] # List of dictionaries for source info

@app.post("/ask", response_model=AskResponse)
async def ask_question(request: AskRequest):
    """
    Answers a question based on the indexed documents, maintaining conversation history.
    """
    try:
        # Convert chat history to LangChain message format
        formatted_chat_history = []
        for human_msg, ai_msg in request.chat_history:
            formatted_chat_history.append(HumanMessage(content=human_msg))
            formatted_chat_history.append(AIMessage(content=ai_msg))

        # Invoke the QA chain
        result = qa_chain.invoke({
            "question": request.query,
            "chat_history": formatted_chat_history
        })

        answer = result["answer"]
        source_docs_info = []
        if "source_documents" in result and result["source_documents"]:
            for doc in result["source_documents"]:
                source_docs_info.append({
                    "page_content": doc.page_content,
                    "metadata": doc.metadata
                })

        return AskResponse(answer=answer, source_documents=source_docs_info)

    except Exception as e:
        print(f"Error processing request: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Basic root endpoint for health check
@app.get("/")
async def read_root():
    return {"message": "LangChain QA API is running. Use /ask to query."}

