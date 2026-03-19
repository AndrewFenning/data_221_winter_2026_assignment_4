# Import necessary libraries
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# Import the 'data' variable from your custom load data file
from data221w26a4LoadData import data

# Construct the feature matrix X and target vector y
X = data.data
y = data.target

print("--- Q2: Decision Tree Model Using Entropy ---")

# 1. 80/20 train-test split with stratification
# random_state ensures we get the exact same split every time we run the code
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)

# 2. Train a Decision Tree classifier using entropy as the splitting criterion
dt_entropy = DecisionTreeClassifier(criterion='entropy', random_state=42)
dt_entropy.fit(X_train, y_train)

# 3. Report the training accuracy and test accuracy
train_acc = dt_entropy.score(X_train, y_train)
test_acc = dt_entropy.score(X_test, y_test)

print(f"Training Accuracy: {train_acc:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")

# --- DISCUSSION COMMENTS ---
# Q: What does entropy represent in the context of decision trees?
# A: Entropy represents the mathematical measure of impurity, uncertainty, or
#    randomness in a specific set of data. In a decision tree, it quantifies how
#    "mixed" the target classes (benign vs. malignant) are within a particular
#    node. The model learns by selecting splits that maximize the reduction in
#    entropy (Information Gain), aiming to create child nodes that are as pure
#    as possible.
#
# Q: Do the observed results suggest overfitting or good generalization?
# A: The observed results suggest overfitting. An unconstrained decision tree
#    typically achieves a training accuracy of exactly 1.0 (100%) because it
#    continues to grow deeper until every leaf perfectly classifies the training
#    samples. It essentially memorizes the training data, including its noise,
#    which results in a lower test accuracy and poor generalization to unseen data.

# Output:
# --- Q2: Decision Tree Model Using Entropy ---
# Training Accuracy: 1.0000
# Test Accuracy: 0.9123