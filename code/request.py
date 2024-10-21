import requests
import json
import pandas as pd

def results(prompt, user_query, model) -> int:
    """Calls the Ollama API to generate a response to a user query
    
    Args:
        prompt (str): The prompt to use
        user_query (str): The user query to generate a response to
        model (str): The model to use
    """
    url = "http://localhost:11434/api/generate"
    headers = {"Content-Type": "application/json"}
    data = {
        "model": model,
        "prompt": f"{prompt} {user_query}\nDistortion:",
        "stream": False, 
        "options": {
            "temperature": 0,
            "num_ctx": 6144,
        },

    }

    res = requests.post(
        url=url, headers=headers, data=json.dumps(data)
    )

    if res.status_code == 200:
        response_text = res.text
        data = json.loads(response_text)
        actual_response = data["response"]
        return actual_response
    else:
        return -1
