# Voice-Powered-QnA-model-using-Langchain
This repository hosts a robust Question Answering (QA) model built with LangChain, leveraging Pinecone for efficient vector storage and OpenAI's LLMs for intelligent response generation. A key feature of this system is its voice input capability, allowing users to interact with the model using spoken queries via a simple web interface.

I've separated the code into two main Python files: ingest.py for data ingestion into Pinecone, and main.py for the FastAPI application that exposes the QA model. I've also included a requirements.txt file.

1. ingest.py (Data Ingestion Script)
This script will preprocess your dataset, chunk it, embed the chunks using OpenAI embeddings, and then store them in Pinecone.

To run ingest.py:

    a. Save the code above as ingest.py.
    b. Create a .env file in the same directory with your API keys and Pinecone environment:
            OPENAI_API_KEY="your_openai_api_key_here"
            PINECONE_API_KEY="your_pinecone_api_key_here"
            PINECONE_ENVIRONMENT="your_pinecone_environment_here" # e.g., "gcp-starter"
    c. Install the necessary libraries (see requirements.txt below).
    d. Run the script from your terminal: python ingest.py

2. main.py (FastAPI Application for QA)
This script sets up a FastAPI application that exposes an /ask endpoint. It uses the Pinecone index as a retriever and an OpenAI LLM for generating answers.

To run main.py:

    a. Save the code above as main.py.
    b. Ensure your .env file is set up correctly as described for ingest.py.
    c. Install the necessary libraries (see requirements.txt below).
    d. Run the FastAPI application using Uvicorn: uvicorn main:app --reload
            main refers to main.py.
            app is the FastAPI instance.
            --reload enables hot-reloading during development.
    e. Access the API documentation at http://127.0.0.1:8000/docs in your browser.

3. requirements.txt
This file lists all the Python packages required for both ingest.py and main.py.

To install dependencies:
    a. Save the content above as requirements.txt.
    b. Run: pip install -r requirements.txt

Then I have created a simple web interface (index.html) that uses the browser's built-in Web Speech API for speech-to-text conversion. The transcribed text will then be sent to your existing FastAPI backend (main.py) to get the answer.

4. index.html (Web Interface with Speech Input)
This HTML file provides a user interface with a microphone button, a text area to display the transcribed query, and an area to show the model's answer.
It also includes the Text-to-Speech (TTS) functionality. As the user speaks the query, the bot will speak the response along with the response display.

How to Use:

    a. Save the HTML: Save the code above as index.html in a folder accessible by your browser.
    b. Ensure FastAPI is Running: Make sure your main.py FastAPI application is running. If you stopped it, restart it from your terminal using:

        Bash
            uvicorn main:app --reload

        (It should be running on http://127.0.0.1:8000 by default).
    c. Open index.html: Open the index.html file in a Google Chrome browser. The Web Speech API has the best support in Chrome.
    d. Grant Microphone Access: When you click the microphone button for the first time, your browser will ask for permission to access your microphone. Grant it.
    e. Speak Your Query: Click the microphone button, wait for it to indicate "Listening...", and then speak your question clearly. Once you stop speaking (or after a short pause), the transcription will appear, and the query will be sent to your FastAPI backend.

This setup will provide a seamless voice-to-text experience for querying your LangChain QA model!

CONVERSATIONAL MEMORY PERSISTENCE
 I have persisted the chat history. I have stored chat_history in a database Firestore associated with a session ID or user ID and then passing this history from the backend to the frontend on load.
 This will ensure that when a user revisits your application, their previous chat history is loaded, providing a continuous conversation experience.

Important Setup for Firestore:
Before you run the updated main.py, you need to set up Firebase Admin SDK.
For local development, you'll typically:
    a. Go to your Firebase project in the Google Cloud Console.
    b. Navigate to Project settings > Service accounts.
    c. Click Generate new private key and download the JSON file.
    d. Save this JSON file (e.g., serviceAccountKey.json) in the same directory as your main.py.
For Canvas Environment, the __app_id, __firebase_config, and __initial_auth_token variables are automatically provided, and the Firestore setup will be handled using these. I will use the Canvas-compatible initialization in the code.

How to Use:
1. Firebase Service Account Key (Local Only):
    a. If running locally (not in the Canvas environment), download your Firebase service account key JSON file and place it in the same directory as main.py.
    b. Make sure the file is named serviceAccountKey.json or update the path in main.py accordingly.
2. Update main.py and index.html: Replace the contents of your existing files with the updated code provided above.
3. Install Firebase Admin SDK: If you haven't already, install it:
    pip install firebase-admin
4. Run ingest.py (Optional, but Recommended): If you haven't done so after the previous update, run python ingest.py to ensure your Pinecone index has the correct metadata (publication_year, first_author) which is crucial for the filtering feature to work correctly.                                                                
5. Run main.py: Start your FastAPI application: uvicorn main:app --reload.                     
6. Open index.html: Open the index.html file in a Google Chrome browser.                         

Testing Persistence:
1. Open index.html in Chrome. Ask a few questions.
2. Observe the "Session ID" displayed. This ID is stored in your browser's localStorage.
3. Close the browser tab completely.
4. Reopen index.html. You should see the previous conversation load automatically based on the session_id       retrieved from localStorage, and the AI will speak the last message or a welcome message if the history was empty.
This setup provides robust conversational memory persistence, making your Q&A model much more practical and user-friendly for repeated interactions!

CONNECTING TO n8n

Once your main.py FastAPI application is running (e.g., at http://localhost:8000), you can easily connect it to n8n using an HTTP Request node.

n8n Workflow Steps:

1. Start Node: Choose a trigger (e.g., "Webhook" to expose an n8n endpoint, or "Manual" for testing).

2. HTTP Request Node:
        a. Authentication: None (or API Key if you add one to your FastAPI app).
        b. Request Method: POST
        c. URL: http://localhost:8000/ask (replace localhost:8000 with your deployed API URL if applicable).
        d. Body Parameters:
                Body Content Type: JSON
                JSON Body:
                JSON

                {
                    "query": "What are the key differences between BERT and DistilBERT?",
                    "chat_history": []
                }
                - You can make query dynamic by referencing an input from a previous node (e.g., {{ $json.query }}   if your trigger provides a query field).
                - chat_history can be built dynamically from previous turns if you're maintaining state in n8n.
        e. Headers: Content-Type: application/json

3. Process Response (Optional): Add a "Set" node or "Code" node to extract the answer from the HTTP Request node's output ({{ $json.answer }}).

4. Trigger Actions (Optional): Based on the extracted answer, you can then add other n8n nodes to:
        a. Send an email (Email Send node).
        b. Log the interaction (e.g., to a Google Sheet or database).
        c. Update a database (e.g., PostgreSQL, MongoDB nodes).
        d. Send a message to a chat platform (e.g., Slack, Discord nodes).

This setup provides a robust and scalable question-answering system using LangChain, Pinecone, and FastAPI, ready for integration with tools like n8n.


KEY FEATURES:

1. Intelligent Q&A: Answers user queries based on a custom knowledge base (demonstrated with the jamescalam/ai-arxiv-chunked dataset).
2. LangChain Framework: Utilizes LangChain for building the RAG pipeline, including document loading, chunking, embedding, retrieval, and conversational AI.
3. Pinecone Vector Database: Stores and retrieves document embeddings for fast and scalable semantic search.
4. OpenAI Integration: Employs OpenAI's embedding models for vector creation and gpt-4o-mini (or similar) for generating concise and relevant answers.
5. Voice Input: A responsive web frontend (index.html) integrates the Web Speech API, enabling users to speak their questions directly. The spoken input is transcribed to text and then processed by the backend.
6. Real Time Transcription display: While the user is speaking, display the interim (live) transcription in the query-input text area. This provides immediate feedback to the user that the system is listening and accurately capturing their words.
7. More Detailed UI Feedback: Providing clear visual cues for different states: "Listening...", "Processing...", "Thinking...", "Speaking...". This reduces user confusion.
8. Edit/Refine Transcribed Query: Speech-to-text isn't always perfect so allowing the user to quickly review and edit the transcribed text in the query-input box before it's sent to the backend. Have added a "Send" button alongside the microphone.
9. Contextual Reranking: After retrieving an initial set of relevant documents from Pinecone, a reranker model (e.g., a cross-encoder from Hugging Face Transformers) scores the relevance of each retrieved document to the query, pushing the most relevant ones to the top. This significantly improves the quality of context provided to the LLM.
10. More Sophisticated Chunking Strategy: Fixed chunk sizes can sometimes cut off context abruptly. So I have explored LangChain's semantic chunking, or document-aware chunking (e.g., splitting by markdown headings, paragraphs, or sections) during our ingestion process (ingest.py).
11. Metadata Filtering & Faceted Search: Allows users to refine their search based on document metadata (e.g., "papers from 2023", "papers by a specific author").
12. Conversational Memory Persistence using Firestore: This will ensure that when a user revisits your application, their previous chat history is loaded, providing a continuous conversation experience.
13. "Was this helpful?" buttons: : It will store this feedback in our Firestore database (perhaps under the chat_sessions document or a separate feedback collection). And analyzes this feedback to identify common failure modes (e.g., queries where answers are consistently marked as "not helpful") and iteratively improves our data, retrieval, or prompting.
14. FastAPI Backend: A lightweight and high-performance API (main.py) exposes the QA functionality, making it easy to integrate with other applications.
15. n8n Integration Ready: Designed for seamless connection with automation platforms like n8n via standard HTTP requests, allowing for advanced workflow orchestration (e.g., triggering emails, logging, or database updates based on Q&A interactions).

PROJECT STRUCTURE:

1. ingest.py: Script for data preprocessing, chunking, embedding, and uploading to Pinecone.
2. main.py: FastAPI application serving the conversational QA endpoint.
3. index.html: Frontend web page for voice and text input, displaying answers and source documents.
4. requirements.txt: Python dependencies.