"""
binary_classification_iris.py

A script for training a logistic regression classifier on the Iris dataset,
evaluating it using multiple classification metrics, and plotting the ROC curve.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_curve, auc, classification_report
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.utils import Bunch

# Step 1: Load and prepare the dataset
iris_data = load_iris()
# Handle potential tuple unpacking issue
if isinstance(iris_data, tuple):
    iris = iris_data[0]
else:
    iris = iris_data
    
X = iris.data
y = iris.target

# For binary classification, use only two classes: setosa vs versicolor
# Class 0: setosa, Class 1: versicolor (skip virginica)
# Filter for classes 0 and 1
X = X[y != 2]
y = y[y != 2]

# Step 2: Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Step 3: Train a logistic regression model
clf = LogisticRegression()
clf.fit(X_train, y_train)

# Step 4: Make predictions and compute evaluation metrics
y_pred = clf.predict(X_test)
y_scores = clf.predict_proba(X_test)[:, 1]  # Probability of class 1

# Basic metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("Classification Report:\n")
print(classification_report(y_test, y_pred, target_names=iris.target_names[:2]))

print(f"Accuracy:  {accuracy:.3f}")
print(f"Precision: {precision:.3f}")
print(f"Recall:    {recall:.3f}")
print(f"F1 Score:  {f1:.3f}")

# Step 5: Compute and plot ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_scores)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'Logistic Regression (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--', label="Random Classifier (AUC = 0.5)")
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.grid(True)
plt.tight_layout()
plt.show()

# -----------------------------
# Discussion (in comments):
# -----------------------------
# - Accuracy: good for balanced datasets, but hides class-specific errors.
# - Precision: important when false positives are costly (e.g., spam filters).
# - Recall: important when false negatives are costly (e.g., cancer detection).
# - F1 Score: balances precision and recall, useful when both errors matter.
# - ROC AUC: useful when evaluating models across all thresholds.

# For this problem:
# - Dataset is balanced (equal samples of class 0 and 1), so accuracy is reasonable.
# - If one class were rare (e.g., fraud), F1 or ROC AUC would be better metrics.
