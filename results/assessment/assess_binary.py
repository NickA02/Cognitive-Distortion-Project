
import os
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

#Make this your results directory
os.chdir('/Users/nicholasalmy/Desktop/Cognitive-Distortion-Project/results/')
eval_paths = [
    'binary/llama3.2-3b/one_shot.csv',
    'binary/llama3.2-3b/zero_shot.csv',
    'binary/llama3.1-8b/one_shot.csv',
    'binary/llama3.1-8b/zero_shot.csv',
]
GOLD_DATA = pd.read_csv('../datasets/test.csv')['binary_ground_truth']

def main(eval_path):
    inference_df = pd.read_csv(eval_path)

    inference_df['Response'] = inference_df['Response'].apply(lambda x: x.lower())
    inference_df['Response'] = inference_df['Response'].apply(lambda x: x.strip(".'"))

    inference_df['Response'] = inference_df['Response'].apply(catch_responses)
    inference_df['gold'] = GOLD_DATA
    results = inference_df['Response']
    inference_df = inference_df[inference_df['Response'] != -1]

    report_scores(inference_df, eval_path)
    return results


def catch_responses(x):
    match x:
        case 'yes':
            return 1
        case 'no':
            return 0
        case 0:
            return 0
        case 1:
            return 1
        case _:
            print(x)
            return -1

def report_scores(inference_df, eval_path):
    print(f"Evaluated Model: {eval_path.split('/')[1]} Prompt: {eval_path.split('/')[2].removesuffix('.csv')}")
    print(f"Accuracy: {accuracy_score(inference_df['gold'], inference_df['Response'])}")
    print(f"F1-Score (Macro): {f1_score(inference_df['gold'], inference_df['Response'], average='macro', pos_label=1)}")
    print(f"F1-Score (Binary): {f1_score(inference_df['gold'], inference_df['Response'], pos_label=1)}")

if __name__ == '__main__':
    results_table = pd.DataFrame()
    results_table['gold'] = GOLD_DATA
    for eval_path in eval_paths:
        results = main(eval_path)
        results_table[eval_path] = results
    results_table.to_csv('assessment/binary_results.csv', index=False)