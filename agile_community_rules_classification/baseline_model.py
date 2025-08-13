import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict, List, Tuple
import pickle
from pathlib import Path

class RuleBasedSimilarityClassifier:
    """Baseline classifier using similarity to positive/negative examples."""
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        self.examples_by_rule = {}
        self.rule_vectors = {}
        
    def fit(self, train_df: pd.DataFrame):
        """Fit the classifier on training data."""
        # Extract examples by rule
        for rule in train_df['rule'].unique():
            rule_data = train_df[train_df['rule'] == rule]
            self.examples_by_rule[rule] = {
                'positive': rule_data[['positive_example_1', 'positive_example_2']].values.flatten(),
                'negative': rule_data[['negative_example_1', 'negative_example_2']].values.flatten()
            }
        
        # Fit TF-IDF vectorizer on all examples
        all_examples = []
        for rule, examples in self.examples_by_rule.items():
            all_examples.extend(examples['positive'])
            all_examples.extend(examples['negative'])
        
        self.vectorizer.fit(all_examples)
        
        # Vectorize examples for each rule
        for rule, examples in self.examples_by_rule.items():
            pos_vectors = self.vectorizer.transform(examples['positive'])
            neg_vectors = self.vectorizer.transform(examples['negative'])
            
            self.rule_vectors[rule] = {
                'positive': pos_vectors,
                'negative': neg_vectors
            }
    
    def predict_proba(self, texts: List[str], rules: List[str]) -> np.ndarray:
        """Predict violation probabilities."""
        # Vectorize input texts
        text_vectors = self.vectorizer.transform(texts)
        
        probabilities = []
        for i, (text, rule) in enumerate(zip(texts, rules)):
            if rule not in self.rule_vectors:
                # Unknown rule - default to 0.5
                probabilities.append(0.5)
                continue
                
            text_vec = text_vectors[i:i+1]
            
            # Calculate similarities to positive and negative examples
            pos_similarities = cosine_similarity(text_vec, self.rule_vectors[rule]['positive']).flatten()
            neg_similarities = cosine_similarity(text_vec, self.rule_vectors[rule]['negative']).flatten()
            
            # Average similarities
            avg_pos_sim = np.mean(pos_similarities)
            avg_neg_sim = np.mean(neg_similarities)
            
            # Simple probability based on relative similarity
            if avg_pos_sim + avg_neg_sim == 0:
                prob = 0.5
            else:
                prob = avg_pos_sim / (avg_pos_sim + avg_neg_sim)
            
            probabilities.append(prob)
        
        return np.array(probabilities)
    
    def save(self, path: str):
        """Save the trained model."""
        with open(path, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def load(cls, path: str):
        """Load a trained model."""
        with open(path, 'rb') as f:
            return pickle.load(f)
