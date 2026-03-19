# Import necessary libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

# Import the 'data' variable from your custom load data file
from data221w26a4LoadData import data

# Construct the feature matrix X and target vector y
X = data.data
y = data.target

print("--- Q4: Neural Network for Binary Classification ---")

# 1. 80/20 train-test split with stratification (same as previous questions)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)

# 2. Standardize the input features
# This scales the features so they have a mean of 0 and a variance of 1
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. Train a neural network with at least one hidden layer
# MLPClassifier automatically uses a sigmoid (logistic) output unit for binary classification.
# hidden_layer_sizes=(16,) means we have one hidden layer with 16 neurons.
# max_iter is set to 1000 to give the network enough epochs to converge.
nn_model = MLPClassifier(
    hidden_layer_sizes=(16,),
    activation='relu',
    solver='adam',
    max_iter=1000,
    random_state=42
)
nn_model.fit(X_train_scaled, y_train)

# 4. Report training accuracy and test accuracy
train_acc = nn_model.score(X_train_scaled, y_train)
test_acc = nn_model.score(X_test_scaled, y_test)

print(f"Neural Network Training Accuracy: {train_acc:.4f}")
print(f"Neural Network Test Accuracy: {test_acc:.4f}\n")

# --- DISCUSSION COMMENTS ---
# Q: Why is feature scaling necessary for neural networks?
# A: Neural networks use gradient descent optimization to update their weights.
#    If features have vastly different scales (e.g., one feature is in the 1000s
#    and another is 0.01), the gradients will be disproportionate. This can cause
#    the network to converge very slowly, bounce around erratically, or get stuck
#    in local minima. Scaling ensures all features contribute equally to the weight
#    updates, making the training process faster, smoother, and more stable.
#
# Q: What does an epoch represent during neural network training?
# A: An epoch represents one complete pass of the entire training dataset through
#    the neural network (including both the forward pass to make predictions and
#    the backward propagation to update weights). Because the model learns iteratively,
#    it typically takes many epochs (multiple full passes over the data) to adjust
#    the weights sufficiently and minimize the error.