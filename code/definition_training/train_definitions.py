import pandas as pd
import json
import sys
import math
from researcher import request, replace_definition, extract_definition
from evaluate import evaluate

def train_definitions(train_prompt: str, val_prompt: str, train_dataset: pd.DataFrame, val_dataset: pd.DataFrame, num_batches=20, val_frequency=5, epochs=1):
    val_list = []
    definition_list = []
    for i in range(epochs):
        definitions, val_f1, histories = train_epoch(
             train_prompt, 
             val_prompt, 
             train_dataset, 
             val_dataset, 
             num_batches, 
             val_frequency, 
        )
        val_list.append(histories)
        definition_list.append(definitions)
        train_prompt = replace_definition(train_prompt, f'<DEFINITION>{definitions}</DEFINITION>')
        sys.stdout.write(f"\nValidation f1 after epoch {i+1}: {val_f1}")
    return definition_list, val_f1, val_list


def train_epoch(train_prompt: str, val_prompt: str, train_dataset: pd.DataFrame, val_dataset: pd.DataFrame, num_batches=20, val_frequency=5):
    """Train the definitions of the chatbot on a dataset of sample questions and responses.
    Args:
        prompt: a string containing the initial prompt to start training with
        train_dataset: a DataFrame of sample questions and responses to train on
        val_dataset: a DataFrame of sample questions and responses to validate on
        num_batches: the number of batches to split the training dataset into
        val_frequency: the number of batches to train on before validating
        epochs: the number of times to iterate over the training dataset
    Returns:
        str - the best prompt with the updated definitions"""
    val_list = []
    #Prepare the training data
    train_dataset.sample(frac=1)
    batches = split_into_batches(train_dataset, num_batches)
    definitions = f'<DEFINITION>{extract_definition(train_prompt)}</DEFINITION>'
    best_definitions = definitions
    best_f1 = evaluate(val_prompt, val_dataset)
    val_list = []

    #Iterate through batches
    for i, batch in enumerate(batches):
        sys.stdout.write(f"\nTraining Batch {i + 1} of {len(batches)}:\n")
        definitions = train_batch(batch, train_prompt)
        definitions = f'<DEFINITION>{definitions}</DEFINITION>'
        train_prompt = replace_definition(train_prompt, definitions)

        if i % val_frequency == 0:
            val_prompt = replace_definition(val_prompt, definitions)
            val_f1 = evaluate(val_prompt, val_dataset)
            val_list.append((i, val_f1, val_prompt))
            sys.stdout.write(f"\nValidation F1 at iteration {i+1}: {val_f1}")
            
            if val_f1 >= best_f1:
                best_definitions = definitions
                best_f1 = val_f1
            else:
                definitions = best_definitions
    sys.stdout.write(f"\nEvaluating Epoch F1 at iteration {i+1}: {val_f1}")
    val_prompt = replace_definition(val_prompt, definitions)
    val_f1 = evaluate(val_prompt, val_dataset)
    val_list.append((i, val_f1, definitions))
    if val_f1 >= best_f1:
                best_definitions = definitions
                best_f1 = val_f1

    return best_definitions, best_f1, val_list

def split_into_batches(dataframe, batch_size):
    num_batches = math.ceil(len(dataframe) / batch_size)
    batches = [
        dataframe[i * batch_size: (i + 1) * batch_size]
        for i in range(num_batches)
    ]
    return batches

def train_batch(batch: pd.DataFrame, initial_prompt: str) -> str:
    """
        Train the chatbot on a batch of sample questions and responses.
        args:
        batch: a DataFrame of sample questions and responses
        chat_histories: a list of chat histories to append to
        returns:
        str - The new definitions for the prompt
    """
    prompt = initial_prompt
    for i, sample_question in enumerate(batch.iterrows()):
        sample_question = sample_question[1]
        sys.stdout.write(f"\rTraining on sample {i + 1} of {len(batch)}")
        response = request(prompt,
            sample_question['Patient Question'],
            sample_question['Dominant Distortion'],
            sample_question['Secondary Distortion (Optional)'],
            sample_question['Distorted part']
        )
        try:
            prompt = replace_definition(prompt, response[-1]['content'])
        except:
            pass
    sys.stdout.write(f"\r\n")

    return extract_definition(prompt)


if __name__ == "__main__":
    train_dataset = pd.read_csv('datasets/train_def.csv')
    val_dataset = pd.read_csv('datasets/val_def.csv')
    train_prompt = open(f'prompts/researcher/researcher.txt','r').read()
    val_prompt = open(f'prompts/multiclass/gemma-defined/zero-shot.txt','r').read()
    definition_list, val_f1, val_list = train_definitions(
        train_prompt=train_prompt,
        val_prompt=val_prompt,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        num_batches=10,
        val_frequency=20,
        epochs=2
    ) 
    with open('definition_list.json', 'w') as f:
        json.dump(definition_list, f, indent=4)

    with open('val_list.json', 'w') as f:
        json.dump(val_list, f, indent=4)

    open(f'prompts/multiclass/gemma-defined/trained-definitions.txt','w').write(definition_list[-1])
    