from PIL import Image
import io
import pytest
import respx
import httpx
from httpx import AsyncClient, Request
from fastapi import FastAPI
from main import app

def create_test_image_bytes():
    # Create a new image using PIL. Here we're creating a small red image.
    img = Image.new('RGB', (100, 100), color = 'red')

    # Convert the image to a byte array.
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='PNG')  # Save image to the byte array as PNG.
    img_byte_arr = img_byte_arr.getvalue()  # Get the byte data

    return img_byte_arr

img_byte_arr = create_test_image_bytes()

@pytest.fixture
def test_app():
    return app

@pytest.mark.asyncio
@respx.mock
async def test_predict(test_app):
    # Create a small red image for testing
    img = Image.new('RGB', (10, 10), color = 'red')
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='JPEG')
    img_byte_arr = img_byte_arr.getvalue()  # This is your image data

    # Mock image URLs
    image_url1 = "https://example.com/image1.jpg"
    image_url2 = "https://example.com/image2.jpg"

    # Setup mock responses for the image URLs
    respx.get(image_url1).respond(content=img_byte_arr)
    respx.get(image_url2).respond(content=img_byte_arr)

    async with AsyncClient(app=test_app, base_url="http://testserver") as ac:
        # Post data to the predict endpoint
        image_data = {
            "imageUrls": [image_url1, image_url2]
        }
        response = await ac.post("/model/api/v1/predict/", json=image_data)

        # Assert the response status code
        assert response.status_code == 200
        response_json = response.json()

        # Check the response content
        assert "status" in response_json
        assert response_json["status"] == "success"
        assert "data" in response_json
        assert isinstance(response_json["data"], list)  # Ensure it is a list

        # Additional detailed checks can be added here for the content of "data"

@pytest.mark.asyncio
@respx.mock
async def test_image_fetch_failure(test_app):
    image_url = "https://example.com/brokenimage.jpg"
    respx.get(image_url).respond(404)  # Simulating a 404 response

    async with AsyncClient(app=test_app, base_url="http://testserver") as ac:
        image_data = {"imageUrls": [image_url]}
        response = await ac.post("/model/api/v1/predict/", json=image_data)

        assert response.status_code == 400  # Assuming your API handles this and converts to 400
        assert "detail" in response.json()


@pytest.mark.asyncio
@respx.mock
async def test_invalid_image_data(test_app):
    image_url = "https://example.com/corruptimage.jpg"
    corrupt_image_data = b'notreallyanimage'
    respx.get(image_url).respond(content=corrupt_image_data)

    async with AsyncClient(app=test_app, base_url="http://testserver") as ac:
        image_data = {"imageUrls": [image_url]}
        response = await ac.post("/model/api/v1/predict/", json=image_data)

        assert response.status_code == 400
        assert "detail" in response.json()


@pytest.mark.asyncio
async def test_empty_image_urls(test_app):
    async with AsyncClient(app=test_app, base_url="http://testserver") as ac:
        image_data = {"imageUrls": []}
        response = await ac.post("/model/api/v1/predict/", json=image_data)
        
        assert response.status_code == 200
        assert response.json()["status"] == "success"
        assert response.json()["data"] == []  # Expecting an empty list as response

@pytest.mark.asyncio
@respx.mock
async def test_response_data_structure(test_app):
    # Assuming img_byte_arr is defined in this test or imported correctly
    image_url = "https://example.com/validimage.jpg"
    respx.get(image_url).respond(content=img_byte_arr)

    async with AsyncClient(app=test_app, base_url="http://testserver") as ac:
        image_data = {"imageUrls": [image_url]}
        response = await ac.post("/model/api/v1/predict/", json=image_data)

        assert response.status_code == 200
        assert "data" in response.json()
        assert isinstance(response.json()["data"], list)

@pytest.mark.asyncio
@respx.mock
async def test_predict_with_real_images(test_app):
    # List of image URLs
    image_urls = [
        "https://firebasestorage.googleapis.com/v0/b/raylabs-26d1c.appspot.com/o/test%2F00028844_021_Effusion_Mass.png?alt=media&token=c75b1607-abcb-4e52-85a7-0955b9e8d63e",
        "https://firebasestorage.googleapis.com/v0/b/raylabs-26d1c.appspot.com/o/test%2F00028888_001_Atelectasis_Consolidation.png?alt=media&token=d4094eee-fb2f-4d62-9679-f7e5fbe098af",
        "https://firebasestorage.googleapis.com/v0/b/raylabs-26d1c.appspot.com/o/test%2F00030541_003_Emphysema.png?alt=media&token=98be9439-d991-40ab-b420-e580cf852979"
    ]

    # Create a small example image for testing
    img_byte_arr = create_test_image_bytes()  # Use the function from previous example

    # Setup mock responses for each image URL
    for url in image_urls:
        respx.get(url).respond(content=img_byte_arr)

    async with AsyncClient(app=test_app, base_url="http://testserver") as ac:
        # Post data to the predict endpoint
        image_data = {"imageUrls": image_urls}
        response = await ac.post("/model/api/v1/predict/", json=image_data)

        # Assert the response status code
        assert response.status_code == 200
        response_json = response.json()

        # Check the response content
        assert "status" in response_json
        assert response_json["status"] == "success"
        assert "data" in response_json
        assert isinstance(response_json["data"], list)  # Ensure it is a list

        # Additional detailed checks can be added here for the content of "data"
