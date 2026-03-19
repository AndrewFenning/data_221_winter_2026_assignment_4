import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# Import the 'data' variable from your custom load data file
from data221w26a4LoadData import data

# Construct the feature matrix X and target vector y
X = data.data
y = data.target

print("--- Q3: Constrained Decision Tree & Interpretability ---")

# 1. 80/20 train-test split with stratification (same as Q2 for fair comparison)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)

# 2. Train a constrained Decision Tree classifier
# We are using max_depth=3 to stop the tree from growing too deep
dt_constrained = DecisionTreeClassifier(
    criterion='entropy',
    max_depth=3,
    random_state=42
)
dt_constrained.fit(X_train, y_train)

# 3. Report the training accuracy and test accuracy
train_acc = dt_constrained.score(X_train, y_train)
test_acc = dt_constrained.score(X_test, y_test)

print(f"Constrained Training Accuracy: {train_acc:.4f}")
print(f"Constrained Test Accuracy: {test_acc:.4f}\n")

# 4. Display the top five most important features
importances = dt_constrained.feature_importances_
# Get the indices of the top 5 features, sorted in descending order
top_5_indices = np.argsort(importances)[::-1][:5]

print("Top 5 Most Important Features:")
for i in top_5_indices:
    feature_name = data.feature_names[i]
    importance_value = importances[i]
    print(f"- {feature_name}: {importance_value:.4f}")

# --- DISCUSSION COMMENTS ---
# Q: How does controlling model complexity affect overfitting?
# A: Unconstrained decision trees can easily overfit training data by memorizing it[cite: 58].
#    By controlling model complexity (e.g., setting a max_depth), we force the tree to be
#    simpler and more generalized[cite: 59, 63]. As seen in the results, the training accuracy
#    drops slightly from 1.0, but the test accuracy often improves or remains stable,
#    indicating that the model is no longer fitting the random noise in the training set.
#
# Q: How does feature importance contribute to the interpretability of decision trees?
# A: Feature importance shows us which specific variables the model relied on the most to
#    make its splits[cite: 64]. Instead of treating the model like a "black box", this allows
#    data scientists and medical professionals to understand the logic behind the predictions
#    and verify that the model is making decisions based on medically relevant features.