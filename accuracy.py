import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import math

# Dummy data (Replace these with your actual values)
ground_truth_conditions = ['flu', 'migraine', 'asthma']
predicted_conditions = ['flu', 'migraine', 'cold']

# 1. Accuracy
accuracy = accuracy_score(ground_truth_conditions, predicted_conditions)
print(f"Accuracy: {accuracy * 100:.2f}%")

# 2. F1-Score for condition classification
f1 = f1_score(ground_truth_conditions, predicted_conditions, average='weighted')
print(f"F1-Score: {f1:.2f}")

# 3. BLEU Score for response quality
references = [["You should rest and stay hydrated".split()]]
candidates = ["You need to rest and drink water".split()]
smooth = SmoothingFunction().method1
bleu = sentence_bleu(references, candidates[0], smoothing_function=smooth)
print(f"BLEU Score: {bleu:.2f}")

# 4. ROUGE-L Score
scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
scores = scorer.score("You need to rest and drink water", "You should rest and stay hydrated")
rouge_l = scores['rougeL'].fmeasure
print(f"ROUGE-L Score: {rouge_l:.2f}")

# 5. Perplexity (Assume loss = 2.91 from test set)
loss = 2.91
perplexity = math.exp(loss)
print(f"Perplexity: {perplexity:.2f}")

# 6. Stage Compliance Rate
total_conversations = 50
compliant_conversations = 48
compliance_rate = (compliant_conversations / total_conversations) * 100
print(f"Stage Compliance Rate: {compliance_rate:.2f}%")

# 7. User Satisfaction Score
user_ratings = [5, 4, 4, 5, 3, 5, 4, 4, 4, 5]
user_satisfaction = sum(user_ratings) / len(user_ratings)
print(f"User Satisfaction Score: {user_satisfaction:.2f}/5")

# 8. Expert Validation
expert_validated = 46
expert_total = 50
expert_validation = (expert_validated / expert_total) * 100
print(f"Expert Validation: {expert_validation:.2f}%")
