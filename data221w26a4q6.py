import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Import the Fashion MNIST data from your custom load data file
from data221w26a4LoadData import X_train, y_train, X_test, y_test

print("--- Q6: Convolutional Neural Network ---")

# 1. Normalize the pixel values to the range [0,1]
X_train_norm = X_train.astype('float32') / 255.0
X_test_norm = X_test.astype('float32') / 255.0

# 2. Reshape the images to include the channel dimension (grayscale = 1 channel)
X_train_reshaped = X_train_norm.reshape(-1, 28, 28, 1)
X_test_reshaped = X_test_norm.reshape(-1, 28, 28, 1)

# 3. Build a CNN with the required layers
model = Sequential([
    # One Conv2D layer
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
    # One MaxPooling2D layer
    MaxPooling2D(pool_size=(2, 2)),
    # Flatten before the dense layer
    Flatten(),
    # One Dense output layer (10 categories for Fashion MNIST)
    Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 4. Train the model for at least 15 epochs
print("Training CNN for 15 epochs (this may take a minute or two)...")
history = model.fit(X_train_reshaped, y_train, epochs=15, validation_split=0.1, verbose=1)

# 5. Report the test accuracy
test_loss, test_acc = model.evaluate(X_test_reshaped, y_test, verbose=0)
print(f"\nCNN Test Accuracy: {test_acc:.4f}")

# --- DISCUSSION COMMENTS ---
# Q: Why are CNNs generally preferred over fully connected networks for image data?
# A: Fully connected networks require images to be flattened into a 1D vector right
#    away, which completely destroys the 2D spatial structure and relationships between
#    neighboring pixels. CNNs preserve this spatial geometry. They also use parameter
#    sharing (a small filter sliding across the image), which drastically reduces the
#    number of weights needed, making the network much more computationally efficient
#    and less prone to overfitting.
#
# Q: What is the convolution layer learning in this task?
# A: The convolution layer automatically learns to detect spatial patterns from the
#    pixel data. It acts as a feature extractor. In this task, the filters in the
#    Conv2D layer are learning low-level visual features like straight edges, curves,
#    and basic textures that are essential for distinguishing between different
#    types of clothing (like the heel of a boot or the strap of a bag).