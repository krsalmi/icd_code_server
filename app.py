import os
import time
from flask import Flask, request, jsonify
import openai 
import bagel
from threading import Lock
from dotenv import load_dotenv
import requests
import json
import logging
from utils import process_multiline_string, extract_documents_based_on_distance, make_json_objects, filter_unique_parent_codes

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

# Initialize global variables
NGROK_API = None
ngrok_lock = Lock()

# Configure OpenAI API Key
openai_key = os.getenv('OPENAI_API_KEY')
if not openai_key:
    logger.error("OpenAI API key not found. Please set the 'OPENAI_API_KEY' environment variable.")
    raise ValueError("OpenAI API key not found.")

# Configure Bagel API Key and Asset ID
BAGEL_API_KEY = os.getenv('BAGEL_API_KEY')
ASSET_ID = "9a9f9cb6-f83d-480b-b0ab-718c274fab18"

# Initialize Bagel Client
try:
    bagel_client = bagel.Client()
    logger.info("Bagel Client initialized successfully.")
except AttributeError as e:
    logger.error(f"Error initializing Bagel Client: {str(e)}")
    raise

client = openai.OpenAI(
    api_key=openai_key,
)


@app.route('/', methods=['GET'])
def home():
    logger.info("Health check accessed.")
    return jsonify({"status": "OK", "message": "Clinical Note Processor is running."}), 200


@app.route('/save_ngrok', methods=['POST'])
def save_ngrok():
    global NGROK_API
    data = request.get_json()
    if not data or 'ngrok_url' not in data:
        logger.error("Missing 'ngrok_url' in request.")
        return jsonify({"error": "Please provide 'ngrok_url' in the JSON body."}), 400

    with ngrok_lock:
        NGROK_API = data['ngrok_url'].rstrip('/')  # Remove trailing slash if any

    return jsonify({"message": f"NGROK_API URL saved as {NGROK_API}."}), 200


@app.route('/summarize', methods=['POST'])
def summarize():
    data = request.get_json()
    if not data or 'clinical_note' not in data:
        return jsonify({"error": "Please provide 'clinical_note' in the JSON body."}), 400

    clinical_note = data['clinical_note']
    logger.info("Summarizing clinical note.")


    try:
        completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
                {
                    "role": "system",
                    "content": "You are a doctor's assist and you are great at summarizing clinical clinical notes. \
                    Try to keep information in about history of present illness, \
                    past medical history and physical exam. The identity of the patient is not important, it is enough to state \
                    the age and gender of the patient. Do not summarize too much, a good length is 300 words. \
                    You can start the summary directly, no need to explicitly state that this is a summary.",
                },
                {
                    "role": "user",
                    "content": clinical_note,
                }
            ]
        )
        summary = completion.choices[0].message.content
        return jsonify({"summary": summary}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/generate', methods=['POST'])
def generate():
    global NGROK_API
    if not NGROK_API:
        return jsonify({"error": "NGROK_API URL not set. Please POST to /save_ngrok first."}), 400

    data = request.get_json()
    if not data or 'clinical note summary' not in data:
        return jsonify({"error": "Please provide 'clinical note summary' in the JSON body."}), 400

    summary = data['clinical note summary']
    generate_url = f"{NGROK_API}/generate"
    logger.info("Generating ICD-10 codes.")

    max_retries = 5  # Define maximum number of retries
    retry_delay = 1  # Define delay between retries in seconds

    payload = {
        "clinical note summary": summary
    }

    for attempt in range(1, max_retries + 1):
        try:
            response = requests.post(generate_url, json=payload)
            logger.info(f"Attempt {attempt}: Received response with status code {response.status_code}.")

            if response.status_code != 200:
                logger.error(f"NGROK_API responded with status code {response.status_code}.")
                return jsonify({"error": f"NGROK_API responded with status code {response.status_code}."}), 502

            data = response.json()

            # Check if the response content is empty or invalid
            if not data or not data.get('validation'):
                logger.warning(f"Attempt {attempt}: Empty validation received. Retrying in {retry_delay} second(s)...")
                if attempt < max_retries:
                    time.sleep(retry_delay)
                    continue  # Retry the request
                else:
                    logger.error("Max retries exceeded with empty validation.")
                    return jsonify({"error": "NGROK_API returned an empty response after multiple attempts."}), 502
            else:
                logger.info("ICD-10 codes generated successfully.")
                return jsonify(data), 200

        except requests.exceptions.RequestException as e:
            logger.error(f"Request to NGROK_API failed on attempt {attempt}: {str(e)}")
            if attempt < max_retries:
                logger.info(f"Retrying in {retry_delay} second(s)...")
                time.sleep(retry_delay)
                continue  # Retry the request
            else:
                logger.error("Max retries exceeded due to request failures.")
                return jsonify({"error": f"NGROK_API request failed after {max_retries} attempts: {str(e)}"}), 502

    # This point is technically unreachable due to the return statements in the loop
    logger.error("Failed to generate ICD-10 codes.")
    return jsonify({"error": "Failed to generate ICD-10 codes."}), 502



@app.route('/rag', methods=['POST'])
def rag():
    data = request.get_json()
    if not data or 'query_text' not in data:
        return jsonify({"error": "Please provide 'query_text' in the JSON body."}), 400

    query_texts = data['query_text']
    query_texts = process_multiline_string(query_texts)
    print("query_texts", query_texts)

    payload = {
      "n_results": 1,
      "include": ["documents", "distances"],
      "query_texts": query_texts,
      "padding": False,
    }

    try:
        logger.info("Getting rag documents from Bagel.")
        response = bagel_client.query_asset(ASSET_ID, payload, BAGEL_API_KEY)
        logger.info("Extracting documents according to distance.")
        documents = extract_documents_based_on_distance(response)
        json_docs = make_json_objects(documents)
        logger.info("Filtering unique parent codes.")
        filtered_json_docs = filter_unique_parent_codes(json_docs)
        return jsonify({"rag_documents": filtered_json_docs}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500



@app.route('/finalize', methods=['POST'])
def finalize():
    data = request.get_json()
    if not data or 'llama_ret' not in data or 'rag_documents' not in data:
        return jsonify({"error": "Please provide 'llama_ret' and 'rag_documents' in the JSON body."}), 400

    llama_ret = data['llama_ret']
    rag_documents = data['rag_documents']

    # Ensure rag_documents is a list of dicts with 'document' keys
    try:
        # Create a JSONL string by converting each document into a JSON string
        documents_text = "\n".join([json.dumps(doc) for doc in rag_documents])
    except (TypeError, KeyError) as e:
        return jsonify({"error": f"Invalid 'rag_documents' format: {str(e)}"}), 400

    template = f"""
    You are a medical coding expert. Validate the following list which includes ICD-10 codes and their descriptions against the provided clinical documents.
    Output the results in the following format:
    - Provide a maximum of 4 ICD-10-CM codes.
    - Format: '[Code]: [Description]'
    - List each code and its description on a new line.
    - Only include the codes and their descriptions—no extra text.
    - Only include the codes that are relevant to the information to validate.
    - If multiple very similar conditions are listed, only include the first one.

    Information to validate:
    {llama_ret}

    Relevant Clinical Documents:
    {documents_text}

    Please verify the accuracy of the ICD-10 codes and provide the correct ones.
    """

    try:
        completion = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system",
                    "content": "You are a medical coding expert.",
                },
                {
                    "role": "user",
                    "content": template,
                }
            ],
            max_tokens=300,
            temperature=0.3,
        )
        validation = completion.choices[0].message.content

        return jsonify({"validation": validation}), 200


    except openai.OpenAIError as e:
        # Handle OpenAI API errors
        return jsonify({"error": f"OpenAI API error: {str(e)}"}), 500
    except Exception as e:
        # Handle other exceptions
        return jsonify({"error": str(e)}), 500
    

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
