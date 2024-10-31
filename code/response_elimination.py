import pandas as pd
from sys import argv
import requests
import json

def results(prompt, conversation_history, model):
    """Calls the local API to generate a response based on the conversation history."""
    url = f"http://localhost:11434/api/generate"
    headers = {"Content-Type": "application/json"}
    full_prompt = prompt + "\n" + "\n".join(conversation_history) + "\nResponse:"
    data = {
        "model": model,
        "prompt": full_prompt,
        "stream": False,
        "options": {
            "temperature": 0,
            "num_ctx": 6144,
        }
    }

    response = requests.post(url, headers=headers, data=json.dumps(data))
    if response.status_code == 200:
        return json.loads(response.text)["response"]
    else:
        return "API Error"

def multiturn_conversation(prompt, initial_query, model, max_turns=5):
    """Handles multiturn conversation by iterating through turns and updating the conversation history."""
    conversation_history = ["User: " + initial_query]
    for _ in range(max_turns):
        response = results(prompt, conversation_history, model)
        conversation_history.append("Assistant: " + response)
        conversation_history.append("User: " + response)  
    return "\n".join(conversation_history)

def main():
    model = argv[1]
    experience = argv[2]
    classification_type = argv[3]
    shot = argv[4]
    if shot == 1:
        shot = "one-shot"
    elif shot == 0:
        shot = "zero-shot"
    elif shot == 1.1:
        shot = "one-shot-traditional"
    elif shot == 1.2:
        shot = "one-shot-experimental"
    elif shot == 2:
        shot = "two-shot"
    elif shot == 3:
        shot = "three-shot"
    else:
        print(f"Currently does not support shot {shot} for {experience} {classification_type} classification")
        exit()
        
        
    try:
        dataset = argv[5]
    except:
        dataset = "datasets/test.csv"   


    if "-" in model:
        model = model.replace("-", ":")
    
    prompt_path = f'prompts/{classification_type}/{shot}.txt'
    prompt = open(prompt_path, 'r').read()
    
    df = pd.read_csv(dataset)
    df = df.head(2)
    df['Full Conversation'] = df['Patient Question'].apply(lambda query: multiturn_conversation(prompt, query, model))
    
    output_file = f'results/{classification_type}/{model.replace(":", "-")}/{shot}.csv'
    df[['Patient Question', 'Full Conversation']].to_csv(output_file, index=False)

if __name__ == "__main__":
    main()
