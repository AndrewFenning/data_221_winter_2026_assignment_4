import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from sklearn.metrics import confusion_matrix

# Import Fashion MNIST data from your custom load file
from data221w26a4LoadData import X_train, y_train, X_test, y_test

print("--- Q7: CNN Error Analysis and Misclassification Study ---")

# 1. Preprocess the data (Same as Q6)
X_train_norm = X_train.astype('float32') / 255.0
X_test_norm = X_test.astype('float32') / 255.0

X_train_reshaped = X_train_norm.reshape(-1, 28, 28, 1)
X_test_reshaped = X_test_norm.reshape(-1, 28, 28, 1)

# 2. Re-build and re-train the CNN (From Q6)
# We need the model to generate predictions for our error analysis!
model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

print("Training CNN for 15 epochs to perform error analysis...")
model.fit(X_train_reshaped, y_train, epochs=15, validation_split=0.1, verbose=1)

# 3. Generate predictions on the test set
print("\nGenerating predictions on the test set...")
y_pred_probs = model.predict(X_test_reshaped)
# Convert probabilities to actual class labels (0 through 9)
y_pred = np.argmax(y_pred_probs, axis=1)

# 4. Compute and display the confusion matrix (Text Only)
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix (10x10):")
print(cm)

# 5. Identify and visualize at least three misclassified images
# These are the 10 categories in the Fashion MNIST dataset
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Find the indices where the prediction does not match the true label
misclassified_indices = np.where(y_pred != y_test)[0]

# Plot the first 3 misclassified images
plt.figure(figsize=(10, 4))
for i in range(3):
    idx = misclassified_indices[i]
    plt.subplot(1, 3, i + 1)
    # Plot the original un-reshaped image
    plt.imshow(X_test[idx], cmap='gray')

    true_label_name = class_names[y_test[idx]]
    pred_label_name = class_names[y_pred[idx]]

    plt.title(f"True: {true_label_name}\nPred: {pred_label_name}")
    plt.axis('off')

plt.tight_layout()
plt.show()

# --- DISCUSSION COMMENTS ---
# Q: Discuss one pattern you observe in the misclassifications.
# A: A common pattern in Fashion MNIST misclassifications is confusing visually
#    similar upper-body clothing items. For example, the model frequently confuses
#    "Shirts", "T-shirts/tops", and "Pullovers" because they share very similar
#    overall shapes, sleeve lengths, and spatial outlines, making them hard to
#    distinguish in a low-resolution 28x28 grayscale format.
#
# Q: Discuss one realistic method to improve the CNN performance.
# A: One realistic method to improve performance is Data Augmentation. By artificially
#    altering the training images (e.g., applying slight rotations, horizontal flips,
#    or slight zooming), we can increase the diversity of the training data. This helps
#    the CNN become more robust and generalize better to unseen images, reducing
#    overfitting. Adding another Conv2D and MaxPooling2D layer could also
#    help the network learn more complex feature representations.

# Output
# --- Q7: CNN Error Analysis and Misclassification Study ---
# ...
# Training CNN for 15 epochs to perform error analysis...
# Epoch 1/15
# 1688/1688 ━━━━━━━━━━━━━━━━━━━━ 9s 5ms/step - accuracy: 0.8407 - loss: 0.4561 - val_accuracy: 0.8808 - val_loss: 0.3421
# ...
# Epoch 15/15
# 1688/1688 ━━━━━━━━━━━━━━━━━━━━ 7s 4ms/step - accuracy: 0.9484 - loss: 0.1450 - val_accuracy: 0.9085 - val_loss: 0.2861
#
# Generating predictions on the test set...
# 313/313 ━━━━━━━━━━━━━━━━━━━━ 1s 2ms/step
#
# Confusion Matrix (10x10):
# [[857   1   8  19   6   3  96   0  10   0]
#  [  4 978   2  10   3   0   2   0   1   0]
#  [ 16   1 798  10  80   0  95   0   0   0]
#  [ 20   4   8 912  25   0  29   0   2   0]
#  [  3   0  31  20 886   0  60   0   0   0]
#  [  0   0   0   0   0 980   0  14   1   5]
#  [105   0  34  23  60   0 773   0   5   0]
#  [  0   0   0   0   0   8   0 977   1  14]
#  [  3   0   1   5   1   9   8   4 969   0]
#  [  0   0   0   0   0  11   1  43   1 944]]

#there is an attached image 'data221w26a4q7myplot.png'