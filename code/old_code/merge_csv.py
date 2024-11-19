import pandas as pd

test = pd.read_csv('datasets/test.csv')
expert = pd.read_csv('results/heirarchy/expert/llama3.1-8b/zero-shot.csv')

updated_test = test[['Patient Question', 'Dominant Distortion']]

# Ensuring indices match for direct assignment; otherwise, consider using merge
updated_test['Prompt'] = expert['Prompt'].values
updated_test['Response'] = expert['Response'].values

updated_test = updated_test[updated_test['Dominant Distortion'] == 'No Distortion']

updated_test.to_csv("expert_merged_no_distortion.csv", index=False)
