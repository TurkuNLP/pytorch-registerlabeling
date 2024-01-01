import requests

headers = {
    "Content-Type": "application/json",
}

params = {
    "key": "AIzaSyCRp0QPGhcxoxzYGY2sjsq2A6Vne94j0i8",
}

json_data = {
    "contents": [
        {
            "parts": [
                {
                    "text": "what is focal loss, as used in machine learning?",
                },
            ],
        },
    ],
}

response = requests.post(
    "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent",
    params=params,
    headers=headers,
    json=json_data,
)


print(response.json())
