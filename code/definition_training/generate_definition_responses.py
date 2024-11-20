import re
import ollama
import pandas as pd
from request import results

def main():
    prompt = open(f'prompts/multiclass/gemma-defined/zero-shot.txt','r').read()
    definitions = pd.read_csv('datasets/definitions.csv')


    for i in range(250, 2100, 50):
        questions = pd.read_csv('datasets/test.csv')
        print(f"Running iteration {i}")
        iteration = definitions.iloc[i]['definition']
        prompt_eval = replace_definition(prompt, iteration)
        questions["Response"] = questions["Patient Question"].apply(lambda x: results(prompt_eval, x, "llama3.2:3b"))
        questions = questions[["Patient Question", "Response"]].rename(columns={"Patient Question": "Prompt", "Response": "Response"})
        questions.to_csv(f"results/iterations/llama3.2-3b/{i}.csv", index=False)


def send_message(prompt, question) -> str:
    message = [{
        'role': 'system',
        'content': prompt + question
        }]
    
    response: str = ollama.chat(
        model="gemma2:27b",
        messages=message,
        stream=False
    )['message']['content']
    print(response)

    return response


def replace_definition(old_text, iteration):
    # Extract the new definition content between the tags
    updated_text = re.sub(r'<DEFINITION>(.*?)</DEFINITION>', iteration, old_text, flags=re.DOTALL)
    return updated_text


if __name__ == "__main__":
    main()