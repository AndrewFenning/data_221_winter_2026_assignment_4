import numpy as np
# Import the 'data' variable from your custom load file
from data221w26a4LoadData import data

# Load the dataset (you already have this part!)

# 1. Construct the feature matrix X and target vector y
X = data.data
y = data.target

# 2. Report the shape of X and y
print("--- Q1: Dataset Exploration ---")
print(f"Shape of feature matrix X: {X.shape}")
print(f"Shape of target vector y: {y.shape}")

# 3. Report the number of samples belonging to each class
unique, counts = np.unique(y, return_counts=True)
print("\nClass distribution:")
for i in range(len(unique)):
    print(f"Class {unique[i]} ({data.target_names[i]}): {counts[i]} samples")

# --- DISCUSSION COMMENTS ---
# Q: Whether the dataset is balanced or imbalanced?
# A: The dataset is slightly imbalanced. There are more benign samples
#    than malignant samples. However, this is considered a mild imbalance
#    compared to real-world datasets where a minority class might make up less than 1% of the data.

# Q: Why is class balance an important consideration for classification models?
# A: Class balance is crucial because models trained on heavily imbalanced datasets
#    can become biased toward the majority class. A model might achieve high overall
#    accuracy by simply predicting the majority class every time, completely failing
#    to identify the minority class. In medical contexts like this, predicting a
#    malignant tumor as benign (false negative) is highly dangerous, so we need a
#    model that learns both classes fairly.