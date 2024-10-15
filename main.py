import os
from typing import List
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl
import torch
import torchvision.transforms as transforms
from PIL import Image
import torch.backends.cudnn as cudnn
from model import DenseNet121
import httpx
from io import BytesIO

app = FastAPI()

# Allow specific origins (in this case, your frontend's origin)
origins = [
    "http://localhost:5173",
    "http://localhost:3000",  # Add other origins if needed
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# CKPT_PATH = 'D:\Academics\Semester 5\Data Science Project\chexnet-api-server\chexnet_epoch_15_auc_0.8144_new.pth'
CKPT_PATH = 'D:\Academics\Semester 5\Data Science Project\chexnet-api-server\chexnet_epoch_17_auc_0.8457.pth'
# CKPT_PATH = 'D:\Academics\Semester 5\Data Science Project\chexnet-api-server\chexnet_epoch_13_auc_0.8179.pth'
N_CLASSES = 14
CLASS_NAMES = ['Atelectasis', 'Consolidation', 'Infiltration', 'Pneumothorax', 'Edema', 'Emphysema', 'Fibrosis', 'Effusion', 'Pneumonia', 'Pleural_Thickening',
'Cardiomegaly', 'Nodule', 'Mass', 'Hernia']

cudnn.benchmark = True

# Initialize and load the model
model = DenseNet121(N_CLASSES).cuda()
model = torch.nn.DataParallel(model).cuda()

if os.path.isfile(CKPT_PATH):
    print("=> loading checkpoint")
    checkpoint = torch.load(CKPT_PATH, weights_only=True)
    model.load_state_dict(checkpoint)
    print("=> loaded checkpoint")
else:
    print("=> no checkpoint found")

# Define the transformations
transformation_pipeline = transforms.Compose([
    transforms.Resize(256),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

model.eval()

@app.get("/")
async def root():
    return {"message": "Welcome to the chexnet API server!"}

class ImageData(BaseModel):
    imageUrls: List[str]

@app.post("/model/api/v1/predict/")
async def predict(data: ImageData):
    predictions = []

    async with httpx.AsyncClient() as client:
        for image_url in data.imageUrls:
            try:
                # Fetch the image from URL
                resp = await client.get(image_url)
                resp.raise_for_status()
            except httpx.RequestError as e:
                raise HTTPException(status_code=400, detail=f"Error fetching image: {str(e)}")
            except httpx.HTTPStatusError as e:
                raise HTTPException(status_code=e.response.status_code, detail="Error fetching image, bad response from server")

            # Open the image and convert to RGB
            img = Image.open(BytesIO(resp.content)).convert('RGB')

            # Assume transformation_pipeline and model are defined elsewhere
            img_tensor = transformation_pipeline(img)
            img_tensor = img_tensor.unsqueeze(0)

            # Predict
            pred = []
            with torch.no_grad():
                output = model(img_tensor)
                values = output.squeeze().tolist()
                prediction = torch.nn.functional.sigmoid(output).squeeze().tolist()

            for i in range(len(CLASS_NAMES)):
                pred.append({"disease": CLASS_NAMES[i], "model_value": values[i], "sigmoid_value": prediction[i]})

            predictions.append({"image_url": image_url, "prediction": pred})

    return {"status": "success", "data": predictions}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
