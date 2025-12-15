
import requests
import json

url = "http://localhost:8000/api/v1/query"
headers = {"Content-Type": "application/json"}
data = {
    "question": "hi",
    "selected_text": None,
    "page_content": None
}

try:
    print(f"Sending POST to {url}...")
    response = requests.post(url, headers=headers, json=data)
    print(f"Status Code: {response.status_code}")
    print("Response JSON:")
    print(json.dumps(response.json(), indent=2))
except Exception as e:
    print(f"Request failed: {e}")
