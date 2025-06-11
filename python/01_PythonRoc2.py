"""
breast_cancer_classification.py

Train a logistic regression classifier on the breast cancer dataset.
Evaluate using classification metrics and plot a realistic ROC curve.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_curve, auc, classification_report
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import Bunch
from typing import cast

# Step 1: Load and inspect the breast cancer dataset
data: Bunch = cast(Bunch, load_breast_cancer())

X = data.data
y = data.target
target_names = data.target_names  # ['malignant', 'benign']

# Step 2: Standardize features (important for models like logistic regression)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 3: Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42, stratify=y
)

# Step 4: Train logistic regression model
clf = LogisticRegression(max_iter=10000, solver='lbfgs')
clf.fit(X_train, y_train)

# Step 5: Predict and evaluate
y_pred = clf.predict(X_test)
y_scores = clf.predict_proba(X_test)[:, 1]  # Probability of positive class

# Step 6: Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("Classification Report:\n")
print(classification_report(y_test, y_pred, target_names=target_names))
print(f"Accuracy:  {accuracy:.3f}")
print(f"Precision: {precision:.3f}")
print(f"Recall:    {recall:.3f}")
print(f"F1 Score:  {f1:.3f}")

# Step 7: Plot ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_scores)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"Logistic Regression (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], "k--", label="Random Classifier (AUC = 0.5)")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve: Breast Cancer Classification")
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.show()

# -----------------------------
# Discussion (in comments):
# -----------------------------
# This dataset has some noise and class overlap, unlike perfectly separable data.
# - Accuracy is high, but not perfect — there's room for model improvement.
# - ROC AUC gives a threshold-independent sense of model quality.
# - Logistic Regression outputs probabilities, enabling ROC analysis.
# - This is a more realistic example for metric tradeoff discussions.
