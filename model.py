from typing import List
from fastapi import HTTPException
import logging
from gradio_client import Client
import httpx
import time  # Import time module for tracking execution time

async def getPredictions(image_urls: List[str]):
    predictions = []
    gradio_client = Client("https://iruda21cse-chextnet-raylabs.hf.space")
    async with httpx.AsyncClient() as client:
        for image_url in image_urls:
            try:
                # Validate the URL by fetching the image
                resp = await client.get(image_url)
                resp.raise_for_status()

                # Start the timer
                start_time = time.time()

                # Send URL to Hugging Face Gradio app
                result = gradio_client.predict(image_url=image_url)

                # End the timer
                end_time = time.time()

                # Calculate the prediction time
                prediction_time = end_time - start_time
            except Exception as e:
                logging.error(f"Error processing image URL: {e}")
                raise HTTPException(status_code=500, detail=f"Internal server error: {e}")

            predictions.append({
                "image_url": image_url,
                "prediction": result,
                "prediction_time": prediction_time  # Include prediction time in the output
            })

    return predictions
