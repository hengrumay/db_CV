# Databricks notebook source
# MAGIC %md
# MAGIC ## Computer Vision: CNN and YOLO

# COMMAND ----------

# MAGIC %md
# MAGIC - **Convolutional Neural Networks** (CNNs) are a class of deep neural networks that are particularly effective for analyzing visual data, widely used for image classification, object detection, image segmentation, and other computer vision tasks.
# MAGIC - **YOLO** (You Only Look Once) is a state-of-the-art, real-time object detection system. It is known for its speed and accuracy in detecting objects within images.

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.GPU configuration
# MAGIC

# COMMAND ----------

cluster_id = dbutils.entry_point.getDbutils().notebook().getContext().tags().get("clusterId").get()
print(f"Cluster ID: {cluster_id}")

# COMMAND ----------

# Get all Spark configuration settings
spark_conf = spark.sparkContext.getConf().getAll()

# Print the Spark configuration settings
for conf in spark_conf:
    print(conf)

# COMMAND ----------

import subprocess

try:
    # Check for GPU using the nvidia-smi command
    result = subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode == 0:
        print("This cluster is using GPU.")
    else:
        print("This cluster is using CPU.")
except FileNotFoundError:
    print("This cluster is using CPU.")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.CNN Training

# COMMAND ----------

# MAGIC %md
# MAGIC - Single Node Training refers to training a Convolutional Neural Network (CNN) on a single machine, typically using the CPU or GPU resources available on that machine. 
# MAGIC - Multi-Node Training refers to distributing the training process across multiple machines (nodes) to accelerate training and handle larger datasets. On Databricks, this is typically achieved using distributed training frameworks like **Horovod** or **TensorFlow’s tf.distribute.Strategy**.

# COMMAND ----------

import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
import matplotlib.pyplot as plt

# Load CIFAR-10 dataset
# The CIFAR-10 dataset contains 60,000 32x32 color images in 10 classes, with 6,000 images per class.
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalize pixel values to be between 0 and 1
# The images are represented as numpy arrays of shape (32, 32, 3) with pixel values ranging from 0 to 255.
# Dividing by 255.0 scales the pixel values to the range [0, 1], which helps in faster convergence during training.
x_train, x_test = x_train / 255.0, x_test / 255.0

# COMMAND ----------

# Function to plot a few images from the dataset
def plot_sample_images(images, labels, class_names, num_images=10):
    plt.figure(figsize=(10, 10))
    for i in range(num_images):
        plt.subplot(5, 5, i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(images[i], cmap=plt.cm.binary)
        # CIFAR-10 labels are integers from 0 to 9, corresponding to 10 classes
        plt.xlabel(class_names[labels[i][0]])
    plt.show()

# Class names for CIFAR-10 dataset
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Plot a few sample images from the training set
plot_sample_images(x_train, y_train, class_names)

# COMMAND ----------

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Initialize the CNN
model = Sequential()

# Add convolutional layer
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.25))

# Add second convolutional layer
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.25))

# Add third convolutional layer
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.25))

# Flatten the layers
model.add(Flatten())

# Add fully connected layer
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))

# Add output layer
model.add(Dense(10, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# COMMAND ----------

# Train the model
model.fit(x_train, y_train, epochs=10, batch_size=64, validation_data=(x_test, y_test))

# COMMAND ----------

# Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f'\nTest accuracy: {test_acc}')

# COMMAND ----------

# Function to plot images with their predicted and true labels
def plot_images(images, true_labels, pred_labels, classes, num_images=5):
    plt.figure(figsize=(15, 5))
    for i in range(num_images):
        plt.subplot(1, num_images, i+1)
        plt.imshow(images[i])
        plt.title(f'True: {classes[true_labels[i][0]]}\nPred: {classes[pred_labels[i]]}')
        plt.axis('off')
    plt.show()

# Review a few predictions
predictions = model.predict(x_test)
pred_labels = np.argmax(predictions, axis=1)

# Define the class labels
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Plot a few images with their predictions
plot_images(x_test[:5], y_test[:5], pred_labels[:5], classes)

# COMMAND ----------

# Save the model
# model.save('/dbfs/tmp/cifar10_cnn_model.h5')
