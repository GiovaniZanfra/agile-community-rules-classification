#!/usr/bin/env python3
"""Train the rule-based similarity baseline classifier."""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
import argparse
from pathlib import Path

from agile_community_rules_classification.data import load_train_data
from agile_community_rules_classification.baseline_model import RuleBasedSimilarityClassifier

def train_baseline_model():
    """Train and evaluate the baseline model."""
    
    # Load data
    print("Loading training data...")
    train_df = load_train_data()
    
    # Split data for evaluation
    train_data, eval_data = train_test_split(
        train_df, test_size=0.2, random_state=42, stratify=train_df['rule_violation']
    )
    
    # Train model
    print("Training baseline model...")
    model = RuleBasedSimilarityClassifier()
    model.fit(train_data)
    
    # Evaluate on validation set
    print("Evaluating model...")
    eval_probs = model.predict_proba(
        eval_data['body'].tolist(), 
        eval_data['rule'].tolist()
    )
    
    # Calculate AUC
    auc = roc_auc_score(eval_data['rule_violation'], eval_probs)
    print(f"Validation AUC: {auc:.4f}")
    
    # Save model
    model_path = Path("models/baseline_similarity_model.pkl")
    model_path.parent.mkdir(exist_ok=True)
    model.save(str(model_path))
    print(f"Model saved to {model_path}")
    
    # Save evaluation results
    eval_results = {
        'auc': auc,
        'n_train': len(train_data),
        'n_eval': len(eval_data),
        'n_rules': len(train_data['rule'].unique())
    }
    
    results_path = Path("models/baseline_eval_results.json")
    import json
    with open(results_path, 'w') as f:
        json.dump(eval_results, f, indent=2)
    
    print(f"Evaluation results saved to {results_path}")
    
    return model, eval_results

if __name__ == "__main__":
    train_baseline_model()
