import os, re
import time
from flask import Flask, request, jsonify
import openai 
import bagel
from dotenv import load_dotenv
import json
import logging
from utils import process_multiline_string, extract_documents_based_on_distance, make_json_objects, filter_unique_parent_codes
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel, PeftConfig


app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

def load_model():
    adapter_path = "adapter_model"
    peft_config = PeftConfig.from_pretrained(adapter_path)
    base_model_name = "bagelnet/Llama-3-8B"

    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0,
        llm_int8_has_fp16_weight=False,
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True
    )

    model = PeftModel.from_pretrained(base_model, adapter_path)
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    return model, tokenizer

model, tokenizer = load_model()

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
        return jsonify({"clinical_note_summary": summary}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/generate', methods=['POST'])
def generate():
    max_length = 150
    data = request.get_json()
    if not data or 'clinical_note_summary' not in data:
        return jsonify({"error": "Please provide 'clinical_note_summary' in the JSON body."}), 400

    clinical_note_summary = data['clinical_note_summary']
    logger.info("Calling fine-tuned model.")
    clinical_note_summary = re.sub(r'\s+', ' ', clinical_note_summary).strip()
    logger.info(f"clinical_note_summary after newline removal: {clinical_note_summary}")

    # Prepare the prompt
    prompt = f"""
You are an expert medical coding assistant.

Task: Analyze the following summary of a clinical note and provide a list of appropriate ICD-10-CM codes that best relate to the medical information mentioned.

Instructions:

-Provide a maximum of 4 ICD-10-CM codes.
-Format: [Code]: [Description]
-List each code and its description on a new line.
-Only include the codes and their descriptions—no extra text.

Clinical Note Summary:
{clinical_note_summary}"""

    # prompt = f"""
    # You are an expert medical coding assistant.

    # Task: Analyze the following summary of a clinical note and provide a list of appropriate ICD-10-CM codes that best relate to the medical information mentioned.

    # Instructions:

    # -Provide a maximum of 4 ICD-10-CM codes.
    # -Format: [Code]: [Description]
    # -List each code and its description on a new line.
    # -Only include the codes and their descriptions—no extra text.

    # Clinical Note Summary:
    # {clinical_note_summary}
    # """

    try:
        attempt = 0
        max_attempts = 10
        response_text = ""

        while attempt < max_attempts:
            attempt += 1

            # Encode the input prompt
            input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)

            # Generate a response
            with torch.no_grad():
                output = model.generate(
                    input_ids,
                    max_length=input_ids.shape[1] + max_length,
                    num_return_sequences=1,
                    # no_repeat_ngram_size=2,
                    temperature=0.6,
                    top_p=0.9,
                    pad_token_id=tokenizer.eos_token_id,
                )

            # Decode and return the response
            response = tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True)
            response_text = response.strip()

            # Check if the response is non-empty
            if response_text:
                break
            else:
                logger.info(f"Empty response received. Retrying... (Attempt {attempt}/{max_attempts})")
                time.sleep(2)  # Wait 2 second before retrying

        if not response_text:
            return jsonify({'error': 'The model returned an empty response after 3 attempts.'}), 500

        return jsonify({'response': response_text})

    except Exception as e:
        return jsonify({"error": str(e)}), 500



@app.route('/rag', methods=['POST'])
def rag():
    data = request.get_json()
    if not data or 'query_text' not in data:
        return jsonify({"error": "Please provide 'query_text' in the JSON body."}), 400

    query_texts = data['query_text']
    if not query_texts:
        return jsonify({"error": "Provided 'query_text' cannot be empty."}), 400
    query_texts = process_multiline_string(query_texts)
    logger.info(f"query_texts: {query_texts}")
    
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
