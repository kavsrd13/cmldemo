import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import mlflow
import mlflow.tensorflow
import pickle

# Define the path to your CIFAR-10 data directory
data_directory = "data/cifar-10-batches-py"

# Unpickle function for CIFAR-10 data
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

# Function to load CIFAR-10 data
def load_cifar10_data(data_dir):
    # Initialize the variables
    train_images = []
    train_labels = []
    
    # Load all the data batches
    for i in range(1, 6):
        data_batch = unpickle(os.path.join(data_dir, f'data_batch_{i}'))
        if i == 1:
            train_images = data_batch[b'data']
            train_labels = data_batch[b'labels']
        else:
            train_images = np.vstack((train_images, data_batch[b'data']))
            train_labels = np.hstack((train_labels, data_batch[b'labels']))
    
    # Load the test batch
    test_batch = unpickle(os.path.join(data_dir, 'test_batch'))
    test_images = test_batch[b'data']
    test_labels = np.array(test_batch[b'labels'])
    
    # Reshape the images
    train_images = train_images.reshape((len(train_images), 3, 32, 32)).transpose(0, 2, 3, 1)
    test_images = test_images.reshape((len(test_images), 3, 32, 32)).transpose(0, 2, 3, 1)
    
    return (train_images, train_labels), (test_images, test_labels)

# Load your data
(train_images, train_labels), (test_images, test_labels) = load_cifar10_data(data_directory)

# Normalize the data
train_images, test_images = train_images / 255.0, test_images / 255.0

# Convert labels to numpy array
train_labels, test_labels = np.array(train_labels), np.array(test_labels)

# Define a simple CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10)
])

# Compile the model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Enable MLflow autologging
mlflow.tensorflow.autolog()

# Train the model
with mlflow.start_run():
    model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels))

    mlflow.register_model(
        model_uri=f"runs:/{run.info.run_id}/model",
        name="cifar10-classifier")


# to load model loaded_model = mlflow.tensorflow.load_model("mlruns/<experiment_id>/<run_id>/artifacts/model")

