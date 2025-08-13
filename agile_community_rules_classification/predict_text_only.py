#!/usr/bin/env python3
"""Generate predictions using the trained text-only baseline model."""

import pandas as pd
import numpy as np
import argparse
from pathlib import Path

from agile_community_rules_classification.data import load_test_data
from agile_community_rules_classification.text_only_baseline import TextOnlyBaselineClassifier

def predict_text_only(model_path: str = "models/text_only_baseline_model.pkl", 
                     output_path: str = "submission_text_only.csv"):
    """Generate predictions using the text-only baseline model."""
    
    # Load model
    print(f"Loading model from {model_path}...")
    model = TextOnlyBaselineClassifier.load(model_path)
    
    # Load test data
    print("Loading test data...")
    test_df = load_test_data()
    
    # Generate predictions
    print("Generating predictions...")
    test_predictions = model.predict_proba(
        test_df['body'].tolist(),
        test_df['rule'].tolist()
    )
    
    # Create submission file
    submission = pd.DataFrame({
        'row_id': test_df['row_id'],
        'rule_violation': test_predictions
    })
    
    # Save predictions
    submission.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")
    
    # Print summary
    print(f"Generated {len(test_predictions)} predictions")
    print(f"Prediction range: {test_predictions.min():.4f} - {test_predictions.max():.4f}")
    print(f"Mean prediction: {test_predictions.mean():.4f}")
    print(f"Std prediction: {test_predictions.std():.4f}")
    
    return submission

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate text-only baseline predictions")
    parser.add_argument("--model", default="models/text_only_baseline_model.pkl", 
                       help="Path to trained model")
    parser.add_argument("--output", default="submission_text_only.csv",
                       help="Output file path")
    
    args = parser.parse_args()
    predict_text_only(args.model, args.output)
