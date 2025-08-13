import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import roc_auc_score
import pickle
from pathlib import Path

class TextOnlyBaselineClassifier:
    """Text-only baseline classifier using TF-IDF + Logistic Regression."""
    
    def __init__(self, max_features=10000):
        self.max_features = max_features
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
        X_text = train_df['body'].fillna('')
        X_rule = train_df['rule']
        y = train_df['rule_violation']
        
        # Create pipeline
        self.pipeline = Pipeline([
            ('features', ColumnTransformer([
                ('text', self.text_vectorizer, 'body'),
                ('rule', 'passthrough', ['rule_encoded'])
            ])),
            ('classifier', self.model)
        ])
        
        # Encode rules
        train_df_copy = train_df.copy()
        train_df_copy['rule_encoded'] = self.rule_encoder.fit_transform(train_df_copy['rule'])
        
        # Fit pipeline
        self.pipeline.fit(train_df_copy[['body', 'rule_encoded']], y)
        
        return self
    
    def predict_proba(self, texts, rules):
        """Predict violation probabilities."""
        # Prepare test data
        test_df = pd.DataFrame({
            'body': texts,
            'rule': rules
        })
        
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
    
    def save(self, path: str):
        """Save the trained model."""
        with open(path, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def load(cls, path: str):
        """Load a trained model."""
        with open(path, 'rb') as f:
            return pickle.load(f)
