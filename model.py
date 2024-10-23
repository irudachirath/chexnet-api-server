import os
from typing import List
from fastapi import HTTPException
from PIL import Image
import logging
from gradio_client import Client, file
import tempfile
import httpx
from io import BytesIO

async def getPedictions(data: List[str]):
    predictions = []
    async with httpx.AsyncClient() as client:
        for image_url in data.imageUrls:
            try:
                # Fetch the image from URL
                resp = await client.get(image_url)
                resp.raise_for_status()
            except Exception as e:
                logging.error(f"Error fetching image from URL: {e}")
                raise HTTPException(status_code=500, detail=f"Internal server error: {e}")

            # Open the image and convert to RGB
            img = Image.open(BytesIO(resp.content)).convert('RGB')

            try:
                # Save the image to a temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
                    img.save(temp_file, format='JPEG')
                    temp_file_path = temp_file.name

                # Send POST request to Hugging Face Gradio app
                gradio_client = Client("https://iruda21cse-chextnet-raylabs.hf.space/")
                result = gradio_client.predict(image=file(temp_file_path))
            except Exception as e:
                logging.error(f"Error sending image to Hugging Face Gradio app: {e}")
                raise HTTPException(status_code=500, detail=f"Internal server error: {e}")
            finally:
                # Clean up the temporary file
                os.remove(temp_file_path)

            predictions.append({"image_url": image_url, "prediction": result})
    return predictions
