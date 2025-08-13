#!/usr/bin/env python3
"""Generate predictions using the trained baseline model."""

import pandas as pd
import numpy as np
import argparse
from pathlib import Path

from agile_community_rules_classification.data import load_test_data
from agile_community_rules_classification.baseline_model import RuleBasedSimilarityClassifier

def predict_baseline(model_path: str = "models/baseline_similarity_model.pkl", 
                    output_path: str = "submission_baseline.csv"):
    """Generate predictions using the baseline model."""
    
    # Load model
    print(f"Loading model from {model_path}...")
    model = RuleBasedSimilarityClassifier.load(model_path)
    
    # Load test data
    print("Loading test data...")
    test_df = load_test_data()
    
    # Generate predictions
    print("Generating predictions...")
    predictions = model.predict_proba(
        test_df['body'].tolist(),
        test_df['rule'].tolist()
    )
    
    # Create submission file
    submission = pd.DataFrame({
        'row_id': test_df['row_id'],
        'rule_violation': predictions
    })
    
    # Save predictions
    submission.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")
    
    # Print summary
    print(f"Generated {len(predictions)} predictions")
    print(f"Prediction range: {predictions.min():.4f} - {predictions.max():.4f}")
    print(f"Mean prediction: {predictions.mean():.4f}")
    
    return submission

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate baseline predictions")
    parser.add_argument("--model", default="models/baseline_similarity_model.pkl", 
                       help="Path to trained model")
    parser.add_argument("--output", default="submission_baseline.csv",
                       help="Output file path")
    
    args = parser.parse_args()
    predict_baseline(args.model, args.output)
