import pandas as pd
from pathlib import Path
from typing import Tuple

def load_train_data() -> pd.DataFrame:
    """Load training data from raw directory."""
    return pd.read_csv("data/raw/train.csv")

def load_test_data() -> pd.DataFrame:
    """Load test data from raw directory."""
    return pd.read_csv("data/raw/test.csv")

def get_examples_by_rule(train_df: pd.DataFrame) -> dict:
    """Extract positive and negative examples for each rule."""
    examples = {}
    for rule in train_df['rule'].unique():
        rule_data = train_df[train_df['rule'] == rule]
        examples[rule] = {
            'positive': rule_data[['positive_example_1', 'positive_example_2']].values.flatten(),
            'negative': rule_data[['negative_example_1', 'negative_example_2']].values.flatten()
        }
    return examples
