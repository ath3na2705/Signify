import tensorflow as tf
from tensorflow.python.keras.layers import Dense, Flatten
from tensorflow.python.keras import Sequential

# Sample model
model = Sequential([
    Flatten(input_shape=(180, 180, 3)),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Create a dummy dataset
import numpy as np

x_train = np.random.rand(100, 180, 180, 3)
y_train = np.random.randint(0, 10, size=(100,))

# Fit the model on dummy data
model.fit(x_train, y_train, epochs=1)