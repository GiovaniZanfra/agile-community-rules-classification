#!/usr/bin/env python3
"""
Reddit Comment Rule Violation Classifier - Text-Only Baseline (Kaggle Submission)
Complete text-only baseline pipeline in a single script for easy copy-paste into Kaggle notebook.

This script implements a text-only classifier using TF-IDF + Logistic Regression
without relying on the provided examples, which should generalize better.

Expected AUC: ~0.82 (more realistic)
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# 1. DATA LOADING
# =============================================================================

print("Loading data...")
train_df = pd.read_csv('/kaggle/input/reddit-comment-rule-violation-classification/train.csv')
test_df = pd.read_csv('/kaggle/input/reddit-comment-rule-violation-classification/test.csv')

print(f"Training data: {train_df.shape}")
print(f"Test data: {test_df.shape}")
print(f"Rules: {train_df['rule'].unique()}")

# =============================================================================
# 2. TEXT-ONLY BASELINE MODEL IMPLEMENTATION
# =============================================================================

class TextOnlyBaselineClassifier:
    """Text-only baseline classifier using TF-IDF + Logistic Regression."""
    
    def __init__(self, max_features=10000):
        self.text_vectorizer = TfidfVectorizer(
            max_features=max_features, 
            stop_words='english',
            ngram_range=(1, 2),  # Use unigrams and bigrams
            min_df=2,  # Minimum document frequency
            max_df=0.95  # Maximum document frequency
        )
        self.rule_encoder = LabelEncoder()
        self.model = LogisticRegression(random_state=42, max_iter=1000)
        self.pipeline = None
        
    def fit(self, train_df):
        """Fit the classifier on training data."""
        # Prepare features
        train_df_copy = train_df.copy()
        train_df_copy['rule_encoded'] = self.rule_encoder.fit_transform(train_df_copy['rule'])
        
        # Create pipeline
        self.pipeline = Pipeline([
            ('features', ColumnTransformer([
                ('text', self.text_vectorizer, 'body'),
                ('rule', 'passthrough', ['rule_encoded'])
            ])),
            ('classifier', self.model)
        ])
        
        # Fit pipeline
        self.pipeline.fit(train_df_copy[['body', 'rule_encoded']], train_df_copy['rule_violation'])
        return self
    
    def predict_proba(self, texts, rules):
        """Predict violation probabilities."""
        # Prepare test data
        test_df = pd.DataFrame({'body': texts, 'rule': rules})
        
        # Encode rules (handle unseen rules)
        try:
            test_df['rule_encoded'] = self.rule_encoder.transform(test_df['rule'])
        except ValueError:
            # Handle unseen rules by using most common rule
            most_common_rule = self.rule_encoder.classes_[0]
            test_df['rule'] = most_common_rule
            test_df['rule_encoded'] = self.rule_encoder.transform(test_df['rule'])
        
        # Predict
        probabilities = self.pipeline.predict_proba(test_df[['body', 'rule_encoded']])
        return probabilities[:, 1]  # Return probability of positive class

# =============================================================================
# 3. MODEL TRAINING AND VALIDATION
# =============================================================================

print("\nTraining and validating model...")

# Split data for validation
train_data, eval_data = train_test_split(
    train_df, test_size=0.2, random_state=42, stratify=train_df['rule_violation']
)

# Train the model
model = TextOnlyBaselineClassifier(max_features=10000)
model.fit(train_data)

# Evaluate on validation set
eval_probs = model.predict_proba(
    eval_data['body'].tolist(), 
    eval_data['rule'].tolist()
)

# Calculate AUC
auc = roc_auc_score(eval_data['rule_violation'], eval_probs)
print(f"Validation AUC: {auc:.4f}")

# =============================================================================
# 4. GENERATE FINAL PREDICTIONS
# =============================================================================

print("\nGenerating final predictions...")

# Retrain on full training data
final_model = TextOnlyBaselineClassifier(max_features=10000)
final_model.fit(train_df)

# Generate predictions on test set
test_predictions = final_model.predict_proba(
    test_df['body'].tolist(),
    test_df['rule'].tolist()
)

# Create submission file
submission = pd.DataFrame({
    'row_id': test_df['row_id'],
    'rule_violation': test_predictions
})

# Save submission
submission.to_csv('submission.csv', index=False)

# =============================================================================
# 5. RESULTS SUMMARY
# =============================================================================

print(f"\n=== RESULTS SUMMARY ===")
print(f"Validation AUC: {auc:.4f}")
print(f"Test predictions: {len(test_predictions)}")
print(f"Prediction range: {test_predictions.min():.4f} - {test_predictions.max():.4f}")
print(f"Mean prediction: {test_predictions.mean():.4f}")
print(f"Submission saved as 'submission.csv'")

# Show sample predictions
print(f"\nSample predictions:")
for i in range(min(3, len(test_df))):
    print(f"  {test_df.iloc[i]['row_id']}: {test_predictions[i]:.4f}")

print("\n=== SUBMISSION READY ===")
print("File 'submission.csv' created successfully!")
print("Upload this file to Kaggle for scoring.")
