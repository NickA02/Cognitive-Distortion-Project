
import pandas as pd
from researcher import send_message_single_turn
from evaluate_helpers import get_f1_macro


def evaluate(prompt: str, dataset: pd.DataFrame) -> float:
    """Evaluate the model on a dataset of sample questions and responses.
    args:
        prompt: the prompt to evaluate
        dataset: a DataFrame of sample questions and responses
    returns:
        float - the Evaluation metric on the dataset (Macro F1)
    """
    dataset['Response'] = dataset['Patient Question']\
        .apply(
            lambda x: send_message_single_turn(prompt + x)
        )
    return get_f1_macro(dataset)

