# Reddit Comment Rule Violation Classifier - Kaggle Submission (Minimal Version)
# Copy-paste this entire script into a single Kaggle notebook cell

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# Load data
train_df = pd.read_csv('/kaggle/input/jigsaw-agile-community-rules/train.csv')
test_df = pd.read_csv('/kaggle/input/jigsaw-agile-community-rules/test.csv')

# Baseline model class
class RuleBasedSimilarityClassifier:
    def __init__(self, max_features=5000):
        self.vectorizer = TfidfVectorizer(max_features=max_features, stop_words='english')
        self.examples_by_rule = {}
        self.rule_vectors = {}
        
    def fit(self, train_df):
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
            self.rule_vectors[rule] = {'positive': pos_vectors, 'negative': neg_vectors}
    
    def predict_proba(self, texts, rules):
        text_vectors = self.vectorizer.transform(texts)
        probabilities = []
        for i, (text, rule) in enumerate(zip(texts, rules)):
            if rule not in self.rule_vectors:
                probabilities.append(0.5)
                continue
            text_vec = text_vectors[i:i+1]
            pos_similarities = cosine_similarity(text_vec, self.rule_vectors[rule]['positive']).flatten()
            neg_similarities = cosine_similarity(text_vec, self.rule_vectors[rule]['negative']).flatten()
            avg_pos_sim = np.mean(pos_similarities)
            avg_neg_sim = np.mean(neg_similarities)
            prob = avg_pos_sim / (avg_pos_sim + avg_neg_sim) if (avg_pos_sim + avg_neg_sim) > 0 else 0.5
            probabilities.append(prob)
        return np.array(probabilities)

# Train and validate
train_data, eval_data = train_test_split(train_df, test_size=0.2, random_state=42, stratify=train_df['rule_violation'])
model = RuleBasedSimilarityClassifier(max_features=5000)
model.fit(train_data)
eval_probs = model.predict_proba(eval_data['body'].tolist(), eval_data['rule'].tolist())
auc = roc_auc_score(eval_data['rule_violation'], eval_probs)
print(f"Validation AUC: {auc:.4f}")

# Generate final predictions
final_model = RuleBasedSimilarityClassifier(max_features=5000)
final_model.fit(train_df)
test_predictions = final_model.predict_proba(test_df['body'].tolist(), test_df['rule'].tolist())

# Create submission
submission = pd.DataFrame({'row_id': test_df['row_id'], 'rule_violation': test_predictions})
submission.to_csv('submission.csv', index=False)

print(f"Test predictions: {len(test_predictions)}")
print(f"Prediction range: {test_predictions.min():.4f} - {test_predictions.max():.4f}")
print("Submission saved as 'submission.csv'")
