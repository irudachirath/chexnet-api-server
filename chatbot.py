import asyncio
from fastapi import HTTPException, BackgroundTasks
import json
from google.oauth2 import service_account
from google.auth.transport.requests import Request
from datetime import datetime
from config import get_service_account_info
import time
from pydantic import BaseModel
import logging
import httpx

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

project = 'boreal-ward-437408-p6'
base_url = "https://generativelanguage.googleapis.com"

initial_instruction = (
    "You are a medical assistant specialized in lung and chest diseases. "
    "Your goal is to answer questions related to the symptoms, diagnosis, treatment, prevention, and complications "
    "of the following conditions: Atelectasis, Cardiomegaly, Effusion, Infiltration, Mass, Nodule, Pneumonia, Pneumothorax, "
    "Consolidation, Edema, Emphysema, Fibrosis, Pleural Thickening, Hernia. "
    "For any other conditions or general health questions or any other prompts, respond with: 'I can only answer questions related to certain lung and chest conditions.' "
    "If the question is about symptoms or involves symptom-based queries, provide a diagnosis or possible explanations within the scope of the listed diseases."
)

# Define the necessary scopes for the API you're using
SCOPES = [
    'https://www.googleapis.com/auth/cloud-platform',
    'https://www.googleapis.com/auth/generative-language'
]

# Get the service account info
service_account_info = get_service_account_info()
service_account_info = json.loads(service_account_info)

# Initialize credentials
credentials = service_account.Credentials.from_service_account_info(service_account_info, scopes=SCOPES)

# Function to get an access token using the service account credentials
def get_access_token():
    # Refresh the token if needed
    if not credentials.valid or credentials.expired:
        credentials.refresh(Request())
    return credentials.token

# Initialize chatbot
def initialize_chatbot():
    global headers
    access_token = get_access_token()
    headers = {
        'Authorization': f'Bearer {access_token}',
        'Content-Type': 'application/json',
        'x-goog-user-project': project
    }

async def ask_medical_chatbot(conversation_history):
    try:
        initialize_chatbot()
        parts = [{"text": initial_instruction}]
        for exchange in conversation_history:
            parts.append({"text": exchange.user})
            if exchange.assistant:
                parts.append({"text": exchange.assistant})

        payload = {
            "contents": [{
                "parts": parts
            }]
        }

        async with httpx.AsyncClient() as client:
            # Start the timer before the request
            start_time = time.time()

            response = await client.post(
                url=f'{base_url}/v1beta/tunedModels/book2modifiedanswers-6vt4z4xscfbz:generateContent',
                headers=headers,
                json=payload,
                timeout=60  # Adjust timeout as needed
            )

            # End the timer after the request
            end_time = time.time()

        if response.status_code == 200:
            response_json = response.json()
            try:
                answer = {
                    "text": response_json['candidates'][0]['content']['parts'][-1]['text'].strip(),
                    "safetyRatings": response_json['candidates'][0]['safetyRatings'],
                    "response_time": end_time - start_time  # Calculate and include response time
                }
                return answer
            except Exception as e:
                logging.error(f"Error parsing response: {str(e)}")
                raise HTTPException(status_code=500, detail=f"Error parsing response: {str(e)}")
        else:
            logging.error(f"Error from external API: {response.text}")
            raise HTTPException(status_code=response.status_code, detail=f"Error from external API: {response.text}")

    except httpx.RequestError as e:
        logging.error(f"Error connecting to the external API: {str(e)}")
        raise HTTPException(status_code=502, detail=f"Error connecting to the external API: {str(e)}")
    except Exception as e:
        logging.error(f"Internal server error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
class Conversation(BaseModel):
    user: str
    assistant: str = None

async def get_condition_details(condition: str):
    logger.info(f"Fetching details for condition: {condition}")
    prompt = (
        f"Please provide a detailed overview of {condition}, including its description, suggested testing, complications, lifestyle changes, and treatment options. "
        "Format the response with clear headings and bullet points."
    )
    conversation_history = [Conversation(user=prompt, assistant="")]
    return await ask_medical_chatbot(conversation_history)

async def generate_medical_report(conditions):
    start_time = time.time()  # Start timing
    logging.info("Starting report generation")
    
    # Initialize chatbot
    initialize_chatbot()
    report = {
        "summary": "",
        "data": []
    }

    # Generate summary asynchronously
    report["summary"] = (await ask_medical_chatbot([Conversation(
        user=f"For this response give me the summary as a paragraph without any headings. Based on the following findings from X-ray images, please provide an overall impression of the patient's condition: {', '.join(conditions)}.",
        assistant=""
    )]))["text"]

    # Prepare tasks for fetching condition details
    tasks = [get_condition_details(condition) for condition in conditions]
    condition_details = await asyncio.gather(*tasks)

    for condition, response in zip(conditions, condition_details):
        report["data"].append({
            "condition": condition,
            "details": response['text']
        })

    end_time = time.time()  # End timing
    report_generation_time = end_time - start_time  # Calculate total duration

    return {
        "status": "success",
        "report": report,
        "timestamp": datetime.now().isoformat(),
        "generation_time_seconds": report_generation_time  # Include generation time in the response
    }