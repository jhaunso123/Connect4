import numpy as np
import os
import random
import pickle
import tensorflow as tf
from tensorflow import keras
from keras import layers

"""## Prepare the dataset"""

# Directory containing the training data files
data_dir = 'training_data'

# Initialize empty lists to hold the data
X = []  # Board states
y = []  # Moves

# Iterate through all the files in the directory
for filename in os.listdir(data_dir):
    if filename.endswith('.pkl'):
        file_path = os.path.join(data_dir, filename)
        
        # Load the training data from the current file
        with open(file_path, 'rb') as f:
            training_data = pickle.load(f)
        
        # Convert the training data into numpy arrays and add to the lists
        for board_state, move in training_data:
            X.append(board_state)
            y.append(move)

# Convert the lists to numpy arrays
X = np.array(X)
y = np.array(y)
print(y)
print(X)
print(f"Number of items in the dataset: {len(X)}")

# Check for unique (X, y) pairs
unique_pairs = set()

for i in range(len(X)):
    # Concatenate X[i] and y[i] and convert to a tuple
    concatenated = tuple(X[i].flatten().tolist() + y[i].flatten().tolist())
    unique_pairs.add(concatenated)

num_unique_pairs = len(unique_pairs)
print(f"Number of unique (X, y) pairs in training_data: {num_unique_pairs}")

# Reshape X to match the input shape expected by the neural network
X = X.reshape(X.shape[0], 6, 7, 1)  # Adding a channel dimension

"""## Train the neural network"""

# Define the neural network architecture
model = keras.Sequential([
    layers.Input(shape=(6, 7, 1)),          # Input layer
    layers.Flatten(),                       # Flatten the input (6x7 board with a single channel)
    layers.Dense(256, activation='relu', use_bias=True),   # Hidden layer with 256 neurons and ReLU activation and bias
    layers.Dense(256, activation='relu'),   # Hidden layer with 256 neurons and ReLU activation
    layers.Dropout(0.25),
    layers.Dense(256, activation='relu'),   # Hidden layer with 256 neurons and ReLU activation
    layers.Dense(256, activation='relu'),   # Hidden layer with 256 neurons and ReLU activation
    layers.Dense(7, activation='softmax')   # Output layer with 7 neurons (one for each column), softmax activation
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X, y, epochs=5000)
# Save the trained model
model.save('neural_network/connect4_model.keras')