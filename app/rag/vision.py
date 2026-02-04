import base64
import requests
import os

OLLAMA_URL = os.getenv("OLLAMA_URL")

def describe_image(image_path):

    try:
        with open(image_path, "rb") as f:
            img_bytes = f.read()

        img_base64 = base64.b64encode(img_bytes).decode()

        payload = {
            "model": "llava",
            "prompt": "Describe this biomedical image precisely.",
            "images": [img_base64]
        }

        response = requests.post(OLLAMA_URL, json=payload)

        return response.json().get("response", "")

    except Exception as e:
        return f"Error: {str(e)}"
