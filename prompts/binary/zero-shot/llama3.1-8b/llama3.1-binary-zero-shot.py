import requests
import json
import pandas as pd

df = pd.read_csv("datasets/test.csv")
prompt = open('prompts/binary/zero-shot/zero-shot.txt','r').read()



def results(prompt, user_query) -> int:
    url = "http://localhost:11434/api/generate"
    headers = {"Content-Type": "application/json"}
    data = {
        "model": "llama3.1:8b",
        "prompt": f"{prompt}{user_query}",
        "stream": False, 
    }

    #Read in prompt

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

df["llama3.2response"] = df["Patient Question"].apply(lambda x: results(prompt, x))
df.to_csv('prompts/binary/results/llama3.1-8b/zero_shot.csv')

