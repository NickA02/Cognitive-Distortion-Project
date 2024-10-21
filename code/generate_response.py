"""Script that takes in system args model, classification-type, and shot, to generate a response"""

from request import results
import pandas as pd
from sys import argv

model = argv[1]
experience = argv[2]
classification_type = argv[3]

try:
    shot = argv[4]
    shot = float(shot)
except:
    print(f"Does not support shot {shot}. Currently only supports 0, 1, 1.1 (Traditional), 1.2 (Experimental), 2, 3")
    exit()

try:
    dataset = argv[5]
except:
    dataset = "datasets/test.csv"

help = """
    Evaluate our models...
    Beware!! This script will overwrite any previous results file for the given model, classification type, and shot.

    Run the script with the following arguments:
        1. Model: The model to use -- Currently only supports {llama3.2-3b, llama3.1-8b}
        2. Experience: The persona of the model -- Currently only supports {expert, baseline}
        2. Classification Type: The classification type to use -- Supports {binary, multiclass, heirarchical}
        3. Shot: The shot to use -- Currently only supports {0, 1}
        4. Dataset: The dataset to use 
    """


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

#if model has a '-' in it, replace it with a ':'
if "-" in model:
    model_for_inference = model.replace("-", ":")
elif ":" in model:
    model_for_inference = model
    model = model.replace(":", "-")

try:
    df = pd.read_csv(f"datasets/{dataset}.csv")
except:
    print("Dataset not found... using test.csv instead")
    df = pd.read_csv("datasets/test.csv")

prompt = open(f'prompts/{classification_type}/{shot}.txt','r').read()

df[f"Response"] = df["Patient Question"].apply(lambda x: results(prompt, x, model_for_inference))

df = df[["Patient Question", "Response"]].rename(columns={"Patient Question": "Prompt", "Response": "Response"})
df.to_csv(f'results/{classification_type}/{model}/{shot}.csv')