# ICD 10 Clinical Note Processor server

This is the backend component for the Mavis AI Bagel project.

## Prerequisites
- Python 3.x
- .env file (contact the project maintainer for this file)

## Setup and Running
Create and activate a virtual environment:
   
   python -m venv .
   source bin/activate
   
Install the required packages:
   
   pip install -r requirements.txt
   
Run the server:
   
   python app.py

## Endpoints

Health Check Endpoint:

Route: '/'
    Method: GET
    Purpose: Provides a simple health check to ensure the Clinical Note Processor is running.

Route: '/summarize'
    Method: POST
    Purpose: Summarizes a clinical note using OpenAI's GPT-4 model. It takes a clinical note as input and returns a concise summary focusing on patient's present situation, past medical history and treatment plan. Currently, this endpoint is not in use, summarization is called directly from the frontend.

Route: '/generate'
    Method: POST
    Purpose: Generates ICD-10 codes and descriptions based on a clinical note summary using a fine-tuned language model. It takes a clinical note summary as input and returns appropriate information.

Route: '/rag'
    Method: POST
    Purpose: Performs a retrieval-augmented generation task using the Bagel API. It takes query text as input, processes it into lines, retrieves relevant documents, and returns filtered results with unique parent codes and distance scores that are over certain thresholds.