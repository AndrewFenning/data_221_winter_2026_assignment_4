# Import necessary libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix

# Import the 'data' variable from your custom load data file
from data221w26a4LoadData import data

# Construct the feature matrix X and target vector y
X = data.data
y = data.target

print("--- Q5: Model Evaluation and Comparison ---")

# 1. Prepare the data (80/20 split, stratification)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)

# Standardize features for the Neural Network
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 2. Re-train the Constrained Decision Tree (from Q3)
dt_constrained = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=42)
dt_constrained.fit(X_train, y_train)

# 3. Re-train the Neural Network (from Q4)
nn_model = MLPClassifier(hidden_layer_sizes=(16,), activation='relu', solver='adam', max_iter=1000, random_state=42)
nn_model.fit(X_train_scaled, y_train)

# 4. Generate Predictions
y_pred_dt = dt_constrained.predict(X_test)
y_pred_nn = nn_model.predict(X_test_scaled)

# 5. Compute and Display Confusion Matrices (Text Only)
cm_dt = confusion_matrix(y_test, y_pred_dt)
print("\nConfusion Matrix - Constrained Decision Tree:")
print(cm_dt)

cm_nn = confusion_matrix(y_test, y_pred_nn)
print("\nConfusion Matrix - Neural Network:")
print(cm_nn)

# --- DISCUSSION COMMENTS ---
# Q: Which model would you prefer for this task?
# A: For a breast cancer diagnostic task, I would prefer the Neural Network.
#    In medical diagnoses, False Negatives (predicting a tumor is benign when
#    it is actually malignant) are the most dangerous type of error. Neural
#    Networks generally achieve higher overall accuracy and often yield fewer
#    False Negatives compared to a shallow Decision Tree. However, if doctors
#    strictly require the model's reasoning to be transparent and verifiable,
#    the Decision Tree would be the better choice despite a potential slight drop in accuracy.
#
# Q: Provide one advantage and one limitation of each model.
# A:
#    Constrained Decision Tree:
#    - Advantage: Highly interpretable. You can visualize the exact threshold
#      and feature used at every split, making it easy to explain to stakeholders.
#    - Limitation: Tends to have lower predictive accuracy on complex, non-linear
#      datasets compared to more advanced models, and can be sensitive to small
#      variations in the data.
#
#    Neural Network:
#    - Advantage: Highly capable of capturing complex, non-linear relationships
#      between features, usually leading to superior accuracy and generalization.
#    - Limitation: Acts as a "black box." It is very difficult to interpret exactly
#      how the model combines the 30 features to arrive at its final prediction.
#      It also strictly requires feature scaling to train effectively.

# Output:
# --- Q5: Model Evaluation and Comparison ---
#
# Confusion Matrix - Constrained Decision Tree:
# [[38  4]
#  [ 2 70]]
#
# Confusion Matrix - Neural Network:
# [[41  1]
#  [ 3 69]]