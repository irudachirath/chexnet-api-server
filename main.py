from typing import List
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from model import getPedictions
from chatbot import ask_medical_chatbot, generate_medical_report
from dotenv import load_dotenv
import os

load_dotenv()

app = FastAPI()

# Allow specific origins (in this case, your frontend's origin)
origins = [
    "http://localhost:5173",
    "http://localhost:5000",
    os.environ.get("FRONTEND_URL"),
    os.environ.get("BACKEND_URL"),
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

N_CLASSES = 14
CLASS_NAMES = ['Atelectasis', 'Consolidation', 'Infiltration', 'Pneumothorax', 'Edema', 'Emphysema', 'Fibrosis', 'Effusion', 'Pneumonia', 'Pleural_Thickening',
'Cardiomegaly', 'Nodule', 'Mass', 'Hernia']

@app.get("/")
async def root():
    return {"message": "Welcome to the chexnet API server!"}

class ImageData(BaseModel):
    imageUrls: List[str]

@app.post("/model/api/v1/predict/")
async def predict(data: ImageData):
    predictions = await getPedictions(data)
    return {"status": "success", "data": predictions}
class Conversation(BaseModel):
    user: str
    assistant: str = None

@app.post("/api/v1/chatbot/chat/")
async def chat(conversation_history: List[Conversation]):
    if not conversation_history:
        raise HTTPException(status_code=400, detail="Conversation history is required")
    # Call the chatbot logic and handle any exceptions properly
    return ask_medical_chatbot(conversation_history)

class PositiveCondition(BaseModel):
    conditions: List[str]

@app.post("/api/v1/chatbot/report/")
async def report(conditions: PositiveCondition):
    if not conditions.conditions:
        raise HTTPException(status_code=400, detail="Conditions are required")
    return generate_medical_report(conditions.conditions)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)