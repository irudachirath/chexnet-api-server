import asyncio
from fastapi import HTTPException, BackgroundTasks
import json
from google.oauth2 import service_account
from google.auth.transport.requests import Request
from datetime import datetime
from config import get_service_account_info
import requests
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

# def ask_medical_chatbot(conversation_history):
#     try:
#         # Refresh token and update headers before making the request
#         initialize_chatbot()
        
#         parts = [{"text": initial_instruction}]
#         # print(conversation_history)
#         for exchange in conversation_history:
#             # print(exchange)
#             parts.append({"text": exchange.user})  # Access user input via dot notation
#             if exchange.assistant:
#                 parts.append({"text": exchange.assistant})  # Access assistant response via dot notation

#         payload = {
#             "contents": [{
#                 "parts": parts
#             }]
#         }

#         # Sending the request to the external API
#         response = requests.post(
#             url=f'{base_url}/v1beta/tunedModels/book2modifiedanswers-6vt4z4xscfbz:generateContent',
#             headers=headers,
#             json=payload
#         )

#         # Check if the response is successful
#         if response.status_code == 200:
#             response_json = response.json()
#             try:
#                 answer = {
#                     "text": response_json['candidates'][0]['content']['parts'][-1]['text'].strip(),
#                     "safetyRatings": response_json['candidates'][0]['safetyRatings'],
#                 }
#                 return answer
#             except Exception as e:
#                 # Parsing error in the response structure, raise an HTTP 500 error
#                 raise HTTPException(status_code=500, detail=f"Error parsing response: {str(e)}")
#         else:
#             # If the response code is not 200, raise an error with the response code and text
#             raise HTTPException(status_code=response.status_code, detail=f"Error from external API: {response.text}")
    
#     except requests.RequestException as e:
#         # Network-related errors or request issues should return an HTTP 502 (Bad Gateway)
#         raise HTTPException(status_code=502, detail=f"Error connecting to the external API: {str(e)}")
    
#     except Exception as e:
#         # Any other unexpected errors should raise a generic HTTP 500 error
#         raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")



# # Helper to fetch condition details using the chatbot LLM
# def get_condition_details(condition: str):
#     prompt = (
#         f"Please provide a detailed overview of {condition}, including its description, suggested testing, complications, lifestyle changes, and treatment options. "
#         "Format the response with clear headings and bullet points."
#     )
    
#     # Use Conversation model instead of a dictionary
#     conversation_history = [Conversation(user=prompt, assistant="")]
    
#     # Pass the conversation model to the chatbot function
#     return ask_medical_chatbot(conversation_history)


# # Generate report logic
# def generate_medical_report(conditions):
#     # Initialize chatbot
#     initialize_chatbot()
    
#     # Initialize the report
#     report = {
#         "summary": "",
#         "data": []
#     }

#     # Add the initial prompt to the report
#     prompt = f"For this response give me the summary as a pharagraph without any headings. Based on the following findings from X-ray images, please provide an overall impression of the patient's condition: {', '.join(conditions)}."
#     overall_summary = [Conversation(user=prompt, assistant="")]
#     response = ask_medical_chatbot(overall_summary)
#     report["summary"] = response["text"]
    
#     for condition in conditions:
#         # Get the details for the condition
#         response = get_condition_details(condition)
        
#         # Extract the response from the chatbot
#         response_text = response['text']
        
#         # Add the response to the report
#         report["data"].append({
#             "condition": condition,
#             "details": response_text
#         })
    
#     return {"status": "success", "report": report, "timestamp": datetime.now().isoformat()}
#     # return {"status": "success", "report": report, "timestamp": datetime.datetime.now().isoformat()}

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
            response = await client.post(
                url=f'{base_url}/v1beta/tunedModels/book2modifiedanswers-6vt4z4xscfbz:generateContent',
                headers=headers,
                json=payload,
                timeout=60  # Adjust timeout as needed
            )

        if response.status_code == 200:
            response_json = response.json()
            try:
                answer = {
                    "text": response_json['candidates'][0]['content']['parts'][-1]['text'].strip(),
                    "safetyRatings": response_json['candidates'][0]['safetyRatings'],
                }
                return answer
            except Exception as e:
                logger.error(f"Error parsing response: {str(e)}")
                raise HTTPException(status_code=500, detail=f"Error parsing response: {str(e)}")
        else:
            logger.error(f"Error from external API: {response.text}")
            raise HTTPException(status_code=response.status_code, detail=f"Error from external API: {response.text}")

    except httpx.RequestError as e:
        logger.error(f"Error connecting to the external API: {str(e)}")
        raise HTTPException(status_code=502, detail=f"Error connecting to the external API: {str(e)}")
    except Exception as e:
        logger.error(f"Internal server error: {str(e)}")
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
    logger.info("Starting report generation")
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

    return {"status": "success", "report": report, "timestamp": datetime.now().isoformat()}