"""Script that takes in system args model, classification-type, and shot, to generate a response"""

from request import results
import pandas as pd
from sys import argv

model = argv[1]
classification_type = argv[2]
shot = argv[3]

help = """
    Evaluate our models...
    Beware!! This script will overwrite any previous results file for the given model, classification type, and shot.

    Run the script with the following arguments:
        1. Model: The model to use -- Currently only supports {llama3.2-3b, llama3.1-8b}
        2. Classification Type: The classification type to use -- Supports {binary, multiclass}
        3. Shot: The shot to use -- Currently only supports {0, 1}
    """

if shot == 1:
    shot = "one-shot"
elif shot == 0:
    shot = "zero-shot"
else:
    print(f"Currently only supports 0 or 1 shot, not {shot}")
    exit()

"if model has a '-' in it, replace it with a ':'"
if "-" in model:
    model_for_inference = model.replace("-", ":")
elif ":" in model:
    model_for_inference = model
    model = model.replace(":", "-")

df = pd.read_csv("datasets/test.csv")
prompt = open(f'prompts/{classification_type}/{shot}.txt','r').read()

df[f"Response"] = df["Patient Question"].apply(lambda x: results(prompt, x, model_for_inference))

df = df[["Patient Question", "Response"]].rename(columns={"Patient Question": "Prompt", "Response": "Response"})
df.to_csv(f'results/{classification_type}/{model}/{shot}.csv')