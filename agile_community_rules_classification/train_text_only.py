#!/usr/bin/env python3
"""Train the text-only baseline classifier."""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import roc_auc_score, classification_report
import argparse
from pathlib import Path
import json

from agile_community_rules_classification.data import load_train_data
from agile_community_rules_classification.text_only_baseline import TextOnlyBaselineClassifier

def train_text_only_model():
    """Train and evaluate the text-only baseline model."""
    
    # Load data
    print("Loading training data...")
    train_df = load_train_data()
    
    # Split data for evaluation
    train_data, eval_data = train_test_split(
        train_df, test_size=0.2, random_state=42, stratify=train_df['rule_violation']
    )
    
    # Train model
    print("Training text-only baseline model...")
    model = TextOnlyBaselineClassifier(max_features=10000)
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
    
    # Cross-validation (simplified)
    print("Running cross-validation...")
    cv_scores = []
    from sklearn.model_selection import StratifiedKFold
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (train_idx, val_idx) in enumerate(cv.split(train_df, train_df['rule_violation'])):
        train_fold = train_df.iloc[train_idx]
        val_fold = train_df.iloc[val_idx]
        
        fold_model = TextOnlyBaselineClassifier(max_features=10000)
        fold_model.fit(train_fold)
        
        val_probs = fold_model.predict_proba(val_fold['body'].tolist(), val_fold['rule'].tolist())
        auc = roc_auc_score(val_fold['rule_violation'], val_probs)
        cv_scores.append(auc)
        print(f"  Fold {fold+1}: AUC = {auc:.4f}")
    
    print(f"CV AUC: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
    
    # Save model
    model_path = Path("models/text_only_baseline_model.pkl")
    model_path.parent.mkdir(exist_ok=True)
    model.save(str(model_path))
    print(f"Model saved to {model_path}")
    
    # Save evaluation results
    eval_results = {
        'validation_auc': auc,
        'cv_mean_auc': np.mean(cv_scores),
        'cv_std_auc': np.std(cv_scores),
        'n_train': len(train_data),
        'n_eval': len(eval_data),
        'n_rules': len(train_data['rule'].unique()),
        'max_features': model.max_features
    }
    
    results_path = Path("models/text_only_eval_results.json")
    with open(results_path, 'w') as f:
        json.dump(eval_results, f, indent=2)
    
    print(f"Evaluation results saved to {results_path}")
    
    return model, eval_results

if __name__ == "__main__":
    train_text_only_model()
