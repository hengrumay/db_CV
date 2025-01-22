# Databricks notebook source
# MAGIC %md
# MAGIC ## Computer Vision: YOLO

# COMMAND ----------

# MAGIC %md
# MAGIC - **YOLO** (You Only Look Once) is a state-of-the-art, real-time object detection system. It is known for its speed and accuracy in detecting objects within images.

# COMMAND ----------

# MAGIC %md
# MAGIC #### 1.YOLOv8

# COMMAND ----------

# MAGIC %md
# MAGIC - **Ultralytics YOLOv8** is a cutting-edge, state-of-the-art (SOTA) model that builds upon the success of previous YOLO versions and introduces new features and improvements to further boost performance and flexibility. YOLOv8 is designed to be fast, accurate, and easy to use, making it an excellent choice for a wide range of object detection and tracking, instance segmentation, image classification and pose estimation tasks.
# MAGIC - **Ultralytics** is a company that focuses on creating advanced AI and machine learning solutions, with a strong emphasis on computer vision applications. The company is widely recognized for its work on YOLO (You Only Look Once) object detection models.

# COMMAND ----------

# MAGIC %pip install ultralytics==8.2.2

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# Import necessary libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
import requests
from PIL import Image
from io import BytesIO

# COMMAND ----------

# Define the image URL
image_url = "https://ultralytics.com/images/bus.jpg"

# Download the image
response = requests.get(image_url)
image = Image.open(BytesIO(response.content))

# Display the image
plt.figure(figsize=(10, 10))
plt.imshow(image)
plt.axis('off')
plt.title('Original Image')
plt.show()

# COMMAND ----------

# Convert the image to an OpenCV format (BGR instead of RGB)
# OpenCV uses BGR by default, whereas most image libraries use RGB
image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

# Load the YOLOv8 model
# Specify the model version to use ('yolov8n.pt' is the nano version, which is lightweight and fast)
model = YOLO("yolov8n.pt")  # You can choose different model versions (e.g., 'yolov8s.pt', 'yolov8m.pt')

# Run inference on the image
# This will detect objects in the image and return the results
results = model(image_cv)

# Iterate over the detection results
for result in results:
    boxes = result.boxes  # Get the bounding boxes for detected objects
    for box in boxes:
        # Extract bounding box coordinates
        x1, y1, x2, y2 = box.xyxy[0]  # Get the coordinates in (x1, y1, x2, y2) format
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  # Convert coordinates to integers
        
        cls = int(box.cls[0])  # Get the class ID of the detected object
        conf = float(box.conf[0])  # Get the confidence score of the detection
        
        class_name = model.names[cls]  # Get the class name from the class ID
        
        # Print the detected object's class, confidence, and bounding box coordinates
        print(f"Class: {class_name}, Confidence: {conf:.2f}, Bounding Box: ({x1}, {y1}) - ({x2}, {y2})")

        # Draw bounding boxes on the image
        # Use a blue rectangle (BGR: (255, 0, 0)) and thickness of 2 pixels
        cv2.rectangle(image_cv, (x1, y1), (x2, y2), (255, 0, 0), 2)
        # Put the class name and confidence score above the bounding box
        cv2.putText(image_cv, f"{class_name} {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

# Convert the image back to RGB format for display
# This is necessary because matplotlib expects images in RGB format
image_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)

# Display the image with bounding boxes
plt.figure(figsize=(10, 10))  # Set the figure size
plt.imshow(image_rgb)  # Show the image
plt.axis('off')  # Hide the axes
plt.title('Detected Objects')  # Set the title of the image
plt.show()  # Display the plot

# COMMAND ----------

# MAGIC %md
# MAGIC #### 2.Finetune YOLOv8

# COMMAND ----------

import ultralytics
# Perform system checks for the Ultralytics YOLO model
# Specify the devices to use (e.g., GPUs 0, 1, 2, and 3)
ultralytics.checks(verbose=True, device=[0,1,2,3])

# COMMAND ----------

import os

# Delete the 'LOCAL_RANK' environment variable if it exists
del os.environ['LOCAL_RANK']

from ultralytics import settings

# Update the settings to disable mlflow
settings.update({'mlflow': False})

# COMMAND ----------

from ultralytics import YOLO

# Disable NCCL P2P to avoid potential issues with multi-GPU training
os.environ["NCCL_P2P_DISABLE"] = "1"

# Load the YOLO model
model = YOLO('yolov8n.pt')

# Train the model on the coco128 dataset for 8 epochs using GPUs 0, 1, 2, and 3
results = model.train(data='coco128.yaml', epochs=8, device=[0,1,2,3])

# COMMAND ----------

# Fine-tune the YOLO model on the coco8 dataset

model.tune(data='coco8.yaml', epochs=8, iterations=4, optimizer='AdamW', plots=False, save=False, val=True, device=[0,1,2,3])

# COMMAND ----------

# Save the fine-tuned model
model.save('yolov8n_finetuned.pt')

# COMMAND ----------

# MAGIC %md 
# MAGIC #### 3.YOLO with Standard Spark udf

# COMMAND ----------

import cv2
import numpy as np
from ultralytics import YOLO
from pyspark.sql.functions import udf
from pyspark.sql.types import ArrayType, StringType, FloatType, StructType, StructField, IntegerType
from PIL import Image
from io import BytesIO

# Define the UDF (User Defined Function) for image detection
def predict_objects(image_bytes):
    # Load the YOLOv8 model inside the UDF
    # Ensure the model is loaded inside the function to avoid potential issues in distributed environments
    model = YOLO("yolov8n.pt")
    
    # Convert image bytes to PIL Image
    # This allows us to handle image data in a format compatible with most image processing libraries
    image = Image.open(BytesIO(image_bytes))
    
    # Convert PIL Image to OpenCV format (BGR)
    # OpenCV uses BGR by default, whereas PIL uses RGB
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    # Run inference on the image using the YOLO model
    # This step detects objects in the image and returns the results
    results = model(image_cv)
    
    # Initialize a list to store detection results
    detections = []
    
    # Iterate over the detection results
    for result in results:
        boxes = result.boxes  # Get the bounding boxes for detected objects
        for box in boxes:
            # Extract bounding box coordinates
            x1, y1, x2, y2 = box.xyxy[0]  # Get the coordinates in (x1, y1, x2, y2) format
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  # Convert coordinates to integers
            
            cls = int(box.cls[0])  # Get the class ID of the detected object
            conf = float(box.conf[0])  # Get the confidence score of the detection
            
            class_name = model.names[cls]  # Get the class name from the class ID
            
            # Append the detection result to the list
            detections.append((class_name, conf, x1, y1, x2, y2))
    
    # Return the list of detections
    return detections

# Define the schema for the UDF return type
# This schema specifies the structure of the detection results
schema = ArrayType(StructType([
    StructField("class_name", StringType(), False),  # Class name of the detected object
    StructField("confidence", FloatType(), False),  # Confidence score of the detection
    StructField("x1", IntegerType(), False),  # x-coordinate of the top-left corner of the bounding box
    StructField("y1", IntegerType(), False),  # y-coordinate of the top-left corner of the bounding box
    StructField("x2", IntegerType(), False),  # x-coordinate of the bottom-right corner of the bounding box
    StructField("y2", IntegerType(), False)  # y-coordinate of the bottom-right corner of the bounding box
]))

# Register the UDF
# This allows us to use the UDF within a Spark DataFrame
predict_objects_udf = udf(predict_objects, schema)

# COMMAND ----------

# Read images as binary files from the specified directory
# The "binaryFile" format allows us to read images as binary data
# "/databricks-datasets/cctvVideos/train_images/" is the directory containing the images
df = spark.read.format("binaryFile").load("/databricks-datasets/cctvVideos/train_images/")

# Apply the UDF to predict objects in the images
# The UDF 'predict_objects_udf' is applied to the 'content' column of the DataFrame
# This will add a new column 'predictions' containing the detection results
df_with_predictions = df.withColumn("predictions", predict_objects_udf(df["content"]))

# Show the results
# Select the 'path' and 'predictions' columns to display
# 'truncate=False' ensures that the full content of the 'predictions' column is displayed without truncation
df_with_predictions.select("path", "predictions").show(truncate=False)


# COMMAND ----------

import matplotlib.pyplot as plt

def draw_bounding_boxes(image_bytes, detections):
    # Convert image bytes to PIL Image
    # This allows us to handle image data in a format compatible with most image processing libraries
    image = Image.open(BytesIO(image_bytes))
    
    # Convert PIL Image to OpenCV format (BGR)
    # OpenCV uses BGR by default, whereas PIL uses RGB
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    # Draw bounding boxes on the image
    for detection in detections:
        # Unpack detection details
        class_name, conf, x1, y1, x2, y2 = detection
        
        # Draw the bounding box on the image
        # Use a blue rectangle (BGR: (255, 0, 0)) and thickness of 2 pixels
        cv2.rectangle(image_cv, (x1, y1), (x2, y2), (255, 0, 0), 2)
        
        # Put the class name and confidence score above the bounding box
        cv2.putText(image_cv, f"{class_name} {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
    
    # Convert the image back to RGB format for display
    # This is necessary because matplotlib expects images in RGB format
    image_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
    
    # Return the processed image
    return image_rgb

# Display the first few images with bounding boxes
# Take the first 5 rows from the DataFrame
rows = df_with_predictions.take(2)

# Iterate over the rows to display images with detected objects
for row in rows:
    image_bytes = row["content"]  # Get the image bytes
    detections = row["predictions"]  # Get the detection results
    image_rgb = draw_bounding_boxes(image_bytes, detections)  # Draw bounding boxes on the image
    
    # Display the image with bounding boxes
    plt.figure(figsize=(10, 10))  # Set the figure size
    plt.imshow(image_rgb)  # Show the image
    plt.axis('off')  # Hide the axes
    plt.title(f"Detected Objects in {row['path']}")  # Set the title of the image
    plt.show()  # Display the plot

# COMMAND ----------

# MAGIC %md
# MAGIC #### 4.YOLO with Pandas udf

# COMMAND ----------

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
from pyspark.sql.functions import pandas_udf, PandasUDFType
from pyspark.sql.types import StructType, StructField, StringType, FloatType, IntegerType, ArrayType
from PIL import Image
from io import BytesIO

# Define the schema for the UDF return type
schema = ArrayType(StructType([
    StructField("class_name", StringType(), False),
    StructField("confidence", FloatType(), False),
    StructField("x1", IntegerType(), False),
    StructField("y1", IntegerType(), False),
    StructField("x2", IntegerType(), False),
    StructField("y2", IntegerType(), False)
]))

# Define the Pandas UDF for image detection
@pandas_udf(schema, functionType=PandasUDFType.SCALAR)
def predict_objects_udf(content_series: pd.Series) -> pd.Series:
    # Load the YOLOv8 model inside the UDF
    model = YOLO("yolov8n.pt")
    
    results_list = []
    
    for image_bytes in content_series:
        # Convert image bytes to PIL Image
        image = Image.open(BytesIO(image_bytes))
        
        # Convert PIL Image to OpenCV format
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Run inference on the image
        results = model(image_cv)
        
        # Extract detection results
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                
                class_name = model.names[cls]
                detections.append((class_name, conf, x1, y1, x2, y2))
        
        results_list.append(detections)
    
    return pd.Series(results_list)


# COMMAND ----------

# Read images as binary files from the specified directory
df = spark.read.format("binaryFile").load("/databricks-datasets/cctvVideos/train_images/")

# Apply the Pandas UDF to predict objects in the images
df_with_predictions = df.withColumn("predictions", predict_objects_udf(df["content"]))

# Show the results
df_with_predictions.select("path", "predictions").show(truncate=False)


# COMMAND ----------

import matplotlib.pyplot as plt

def draw_bounding_boxes(image_bytes, detections):
    # Convert image bytes to PIL Image
    image = Image.open(BytesIO(image_bytes))
    
    # Convert PIL Image to OpenCV format
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    # Draw bounding boxes on the image
    for detection in detections:
        class_name, conf, x1, y1, x2, y2 = detection
        cv2.rectangle(image_cv, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(image_cv, f"{class_name} {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
    
    # Convert the image back to RGB format for display
    image_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
    return image_rgb

# Display the first few images with bounding boxes
rows = df_with_predictions.take(2)
for row in rows:
    image_bytes = row["content"]
    detections = row["predictions"]
    image_rgb = draw_bounding_boxes(image_bytes, detections)
    
    plt.figure(figsize=(10, 10))
    plt.imshow(image_rgb)
    plt.axis('off')
    plt.title(f"Detected Objects in {row['path']}")
    plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC #### 5.Yolov8 Model Exploration

# COMMAND ----------

# Load pre-trained segmentation model
model_nano = YOLO('yolov8n-seg.pt')  # nano model
# or 
model_small = YOLO('yolov8s-seg.pt')  # small model
# or
model_medium = YOLO('yolov8m-seg.pt')  # medium model

# COMMAND ----------

plt.imshow(image)
plt.axis('off')
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 6.YOLO Image Segmentation
# MAGIC
# MAGIC - YOLO can also perform segmentation, which involves identifying and delineating the boundaries of objects within an image.
# MAGIC - The segmentation masks generated by YOLO can be used for various applications such as image editing, object tracking, and more.

# COMMAND ----------

results = model_medium(image_cv)

# COMMAND ----------

results

# COMMAND ----------

# Visualize results
results[0].plot()

# Access segmentation masks
masks = results[0].masks

# Convert the result to an image
result_image = Image.fromarray(results[0].plot(show=False))

# Display the image
plt.imshow(result_image)
plt.axis('off')
plt.show()
