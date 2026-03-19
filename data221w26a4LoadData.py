from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()

from tensorflow.keras.datasets import fashion_mnist
( X_train , y_train ) , ( X_test , y_test ) = fashion_mnist.load_data ()
