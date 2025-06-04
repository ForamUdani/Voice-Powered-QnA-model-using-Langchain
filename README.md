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
