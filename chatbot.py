from fastapi import HTTPException
import json
from google.oauth2 import service_account
from google.auth.transport.requests import Request
from datetime import datetime
from config import get_service_account_info
import requests
from pydantic import BaseModel

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

def ask_medical_chatbot(conversation_history):
    try:
        # Refresh token and update headers before making the request
        initialize_chatbot()
        
        parts = [{"text": initial_instruction}]
        # print(conversation_history)
        for exchange in conversation_history:
            # print(exchange)
            parts.append({"text": exchange.user})  # Access user input via dot notation
            if exchange.assistant:
                parts.append({"text": exchange.assistant})  # Access assistant response via dot notation

        payload = {
            "contents": [{
                "parts": parts
            }]
        }

        # Sending the request to the external API
        response = requests.post(
            url=f'{base_url}/v1beta/tunedModels/book2modifiedanswers-6vt4z4xscfbz:generateContent',
            headers=headers,
            json=payload
        )

        # Check if the response is successful
        if response.status_code == 200:
            response_json = response.json()
            try:
                answer = {
                    "text": response_json['candidates'][0]['content']['parts'][-1]['text'].strip(),
                    "safetyRatings": response_json['candidates'][0]['safetyRatings'],
                }
                return answer
            except Exception as e:
                # Parsing error in the response structure, raise an HTTP 500 error
                raise HTTPException(status_code=500, detail=f"Error parsing response: {str(e)}")
        else:
            # If the response code is not 200, raise an error with the response code and text
            raise HTTPException(status_code=response.status_code, detail=f"Error from external API: {response.text}")
    
    except requests.RequestException as e:
        # Network-related errors or request issues should return an HTTP 502 (Bad Gateway)
        raise HTTPException(status_code=502, detail=f"Error connecting to the external API: {str(e)}")
    
    except Exception as e:
        # Any other unexpected errors should raise a generic HTTP 500 error
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

class Conversation(BaseModel):
    user: str
    assistant: str = None

# Helper to fetch condition details using the chatbot LLM
def get_condition_details(condition: str):
    prompt = (
        f"Please provide a detailed overview of {condition}, including its description, suggested testing, complications, lifestyle changes, and treatment options. "
        "Format the response with clear headings and bullet points."
    )
    
    # Use Conversation model instead of a dictionary
    conversation_history = [Conversation(user=prompt, assistant="")]
    
    # Pass the conversation model to the chatbot function
    return ask_medical_chatbot(conversation_history)


# Generate report logic
def generate_medical_report(conditions):
    # Initialize chatbot
    initialize_chatbot()
    
    # Initialize the report
    report = []
    
    for condition in conditions:
        # Get the details for the condition
        response = get_condition_details(condition)
        
        # Extract the response from the chatbot
        response_text = response['text']
        
        # Add the response to the report
        report.append({
            "condition": condition,
            "details": response_text
        })
    
    return {"status": "success", "report": report, "timestamp": datetime.now().isoformat()}
    # return {"status": "success", "report": report, "timestamp": datetime.datetime.now().isoformat()}
