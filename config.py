import os
import base64
from dotenv import load_dotenv

load_dotenv()

def get_service_account_info():
    encoded_key = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS_JSON")
    if not encoded_key:
        raise Exception("Service account key not found in environment variables.")

    # Decode the base64-encoded key
    decoded_key = base64.b64decode(encoded_key).decode('utf-8')
    return decoded_key
