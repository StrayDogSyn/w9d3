"""
synthetic_classification_roc.py

Generate a synthetic binary classification dataset that is moderately predictable.
Train a logistic regression classifier, evaluate its performance, and plot the ROC curve.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_curve, auc, classification_report
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Step 1: Generate synthetic dataset with 75-80% separability
X, y = make_classification(
    n_samples=1000,
    n_features=10,
    n_informative=5,
    n_redundant=2,
    n_clusters_per_class=2,
    flip_y=0.05,              # Add noise
    class_sep=1.0,            # Lower = harder to separate
    random_state=42
)

# Step 2: Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 3: Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42, stratify=y
)

# Step 4: Train a logistic regression classifier
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)

# Step 5: Predictions and metrics
y_pred = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)[:, 1]  # Scores for ROC

# Step 6: Evaluate classification metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("Classification Report:\n")
print(classification_report(y_test, y_pred))
print(f"Accuracy:  {accuracy:.3f}")
print(f"Precision: {precision:.3f}")
print(f"Recall:    {recall:.3f}")
print(f"F1 Score:  {f1:.3f}")

# Step 7: Plot ROC curve
fpr, tpr, _ = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], 'k--', label="Random Classifier (AUC = 0.5)")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve: Logistic Regression on Synthetic Data")
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.show()

# -------------------------------
# Discussion:
# -------------------------------
# With class_sep=1.0 and flip_y=0.05, we simulate a noisy but learnable binary problem.
# You can tune `class_sep` and `flip_y` to adjust how predictable the data is.
# This setting typically yields 75-80% accuracy.
# ROC AUC gives a good threshold-independent measure of performance.
