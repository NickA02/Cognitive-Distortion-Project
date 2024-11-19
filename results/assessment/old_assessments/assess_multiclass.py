
import os
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

#Make this your results directory
os.chdir('/Users/nicholasalmy/Desktop/Cognitive-Distortion-Project/results/')
eval_paths = [
    'multiclass/llama3.2-3b/one_shot.csv',
    'multiclass/llama3.2-3b/zero_shot.csv',
    'multiclass/llama3.1-8b/one_shot.csv',
    'multiclass/llama3.1-8b/zero_shot.csv',
]


def main(eval_path):
    inference_df = pd.read_csv(eval_path)

    inference_df['Response'] = inference_df['Response'].apply(lambda x: x.lower())
    inference_df['Response'] = inference_df['Response'].apply(lambda x: x.strip(".'\n"))

    inference_df['Response'] = inference_df['Response'].apply(catch_responses)
    inference_df['gold'] = GOLD_DATA
    results = inference_df['Response']
    inference_df = inference_df[inference_df['Response'] != -1]

    report_scores(inference_df, eval_path)
    return results


def catch_responses(x):
    match x:
        case 'no distortion': 
            return 0
        case 'emotional reasoning': 
            return 1
        case 'overgeneralization': 
            return 2
        case 'mental filter':
            return 3
        case 'should statements':
            return 4
        case 'all-or-nothing thinking':
            return 5
        case 'mind reading': 
            return 6
        case 'fortune-telling':
            return 7
        case 'magnification': 
            return 8
        case 'catastrophizing':
            return 8
        case 'personalization': 
            return 9
        case 'labeling': 
            return 10
        case _:
            print(x)
            return -1


def report_scores(inference_df, eval_path):
    print(f"Evaluated Model: {eval_path.split('/')[1]} Prompt: {eval_path.split('/')[2].removesuffix('.csv')}")
    print(f"Accuracy: {accuracy_score(inference_df['gold'], inference_df['Response'])}")
    f1_macro = f1_score(inference_df['gold'], inference_df['Response'], average=None)
    for i, f1 in enumerate(f1_macro):
        print(f"F1-Score (Class {i}): {f1}")
    print(f"F1-Score (Macro): {f1_score(inference_df['gold'], inference_df['Response'], average='macro')}")
    print(f"F1-Score (Weighted): {f1_score(inference_df['gold'], inference_df['Response'], average='weighted')}")

GOLD_DATA = pd.read_csv('../datasets/test.csv')['Dominant Distortion']\
    .apply(lambda x: x.lower())\
    .apply(catch_responses)

if __name__ == '__main__':
    results_table = pd.DataFrame()
    results_table['gold'] = GOLD_DATA
    for eval_path in eval_paths:
        results = main(eval_path)
        results_table[eval_path] = results
    #results_table.to_csv('assessment/multiclass_results_test.csv', index=False)