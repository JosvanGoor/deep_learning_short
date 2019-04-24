import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

class_names = \
[
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle Boot"
]

print(tf.__version__)

fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

train_images = train_images / 255.0
test_images = test_images / 255.0

model = keras.Sequential \
(
    [
        keras.layers.Reshape((28, 28, 1), input_shape = (28, 28)),
        keras.layers.Conv2D(32, kernel_size = 2, activation = "relu", padding = "same"),
        keras.layers.MaxPool2D(1),
        keras.layers.Conv2D(64, kernel_size = 4, activation = "relu", padding = "same"),
        keras.layers.MaxPool2D(1),
        keras.layers.Flatten(),
        keras.layers.Dense(10, activation="softmax")
    ]
)

model.compile \
(
    optimizer = "adam",
    loss = "sparse_categorical_crossentropy",
    metrics = ["accuracy"]
)

model.fit(train_images, train_labels, epochs = 5)

test_loss, test_acc = model.evaluate(test_images, test_labels)
print("\nTest accuracy: {}".format(test_acc))