

import pandas as pd
import re

from sklearn.metrics import f1_score


distortions = {
    'n': 0,
    'none': 0,
    'no distortion': 0,
    'emotional reasoning': 1,
    'overgeneralization': 2,
    'mental filter': 3,
    'should statements': 4,
    'all-or-nothing thinking': 5,
    'mind reading': 6,
    'fortune-telling': 7,
    'magnification': 8,
    'personalization': 9,
    'labeling': 10,
}

patterns = {
    'n': r'\bn\b',
    'none': r'\bnone\b',
    'no distortion': r'\bno\s+distortion\b',
    'emotional reasoning': r'\bemotional[\s-]*reasoning\b',
    'overgeneralization': r'\bover[\s-]*generalization\b',
    'mental filter': r'\bmental[\s-]*filter\b',
    'should statements': r'\bshould[\s-]*statements?\b', 
    'all-or-nothing thinking': r'\ball[\s-]*or[\s-]*nothing[\s-]*thinking\b',
    'mind reading': r'\bmind[\s-]*reading\b',
    'fortune-telling': r'\bfortune[\s-]*telling\b',
    'magnification': r'\bmagnification\b',
    'personalization': r'\bpersonalization\b',
    'labeling': r'\blab[e]*ling\b'  
}

def catch_responses(x):
    # Convert to lowercase and normalize whitespace
    normalized_text = re.sub(r'\s+', ' ', x.lower().strip())
    
    # Handle potential asterisks or other formatting
    normalized_text = re.sub(r'[*_]', '', normalized_text)
    
    for label, pattern in patterns.items():
        if re.search(pattern, normalized_text):
            return distortions[label]
    return -1


def remove_punctuation(text):
    return re.sub(r'[^\w\s]', '', text)


def match_either(row):
    if row['Response'] == -1:
        row['gold'] = row['gold_dominant']
        return row
    if row['Response'] == row['gold_secondary']:
        row['gold'] = row['gold_secondary']
    else:
        row['gold'] = row['gold_dominant']
    return row

def get_f1_macro(inference_df: pd.DataFrame) -> float:
    inference_df['Response'] = inference_df['Response']\
        .apply(lambda x: x.lower())\
        .apply(remove_punctuation)\
        .apply(catch_responses)
    
    gold_dominant_data = inference_df['Dominant Distortion']\
    .apply(lambda x: x.lower())\
    .apply(catch_responses)

    gold_secondary_data = inference_df('datasets/test.csv')['Secondary Distortion (Optional)']\
    .apply(lambda x: x if x is None else str(x).lower())\
    .apply(catch_responses)

    inference_df['gold_dominant'] = gold_dominant_data
    inference_df['gold_secondary'] = gold_secondary_data

    inference_df = inference_df.apply(match_either, axis=1)

    return f1_score(inference_df['gold'], inference_df['Response'], labels=range(0,11), average='macro')
