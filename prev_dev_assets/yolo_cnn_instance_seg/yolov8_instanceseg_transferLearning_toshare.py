# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC ## SIMPLE Example of Transfer Learning With Your Data 
# MAGIC
# MAGIC NB:
# MAGIC - With more complex data and multiple labels to train -- the training + tunining process etc. will likely be more involved 
# MAGIC - Data used will also have to be formatted in the required structure that the model/DL framework accepts -- ref e.g. https://docs.ultralytics.com/datasets/segment/#how-do-i-prepare-a-yaml-file-for-training-ultralytics-yolo-models

# COMMAND ----------

# DBTITLE 1,cluster info.
# # Cluster Info. -- derived from cluster yaml 
# "autotermination_minutes": 120,
# "enable_elastic_disk": true,
# "single_user_name": "may.merkletan@databricks.com",
# "enable_local_disk_encryption": false,
# "data_security_mode": "SINGLE_USER",
# "runtime_engine": "STANDARD",
# "effective_spark_version": "14.3.x-cpu-ml-scala2.12",
# "assigned_principal": "user:may.merkletan@databricks.com",
# "autoscale": {
#               "min_workers": 2,
#               "max_workers": 8
#               }
       

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC EXTERNAL Refs: 
# MAGIC - [YOLOv8 instance segmentation transfer learning example](https://towardsdatascience.com/trian-yolov8-instance-segmentation-on-your-data-6ffa04b2debd) 
# MAGIC - [Databricks MLflow integration snippets](https://benhay.es/posts/object_detection_yolov8/) 
# MAGIC - [Other interesting ref on image labelling for segmentation](https://forum.image.sc/t/creating-label-images-with-the-segment-anything-model-guide/93224/20)
# MAGIC
# MAGIC ---    
# MAGIC
# MAGIC **Data Formats / Annotation / Labelling Best Practice**
# MAGIC - https://docs.voxel51.com/integrations/index.html
# MAGIC - https://cocodataset.org/?ref=labellerr.com#format-data 
# MAGIC
# MAGIC WRT Ultralytics formatting:
# MAGIC - https://docs.ultralytics.com/datasets/segment/
# MAGIC - https://docs.ultralytics.com/datasets/segment/#ultralytics-yolo-format
# MAGIC - https://docs.ultralytics.com/guides/data-collection-and-annotation 
# MAGIC
# MAGIC
# MAGIC Example with YOLOv11 
# MAGIC - https://blog.roboflow.com/train-yolov11-instance-segmentation/ | [github_nb](https://github.com/roboflow/notebooks/blob/main/notebooks/train-yolo11-instance-segmentation-on-custom-dataset.ipynb?ref=blog.roboflow.com)

# COMMAND ----------

# DBTITLE 1,Install Dependencies for Example
# MAGIC %pip install scikit-image rasterio ultralytics torchvision 
# MAGIC
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Import Modules/Libraries
import numpy as np
from PIL import Image
from skimage import draw # from scikit-image package
import random
from pathlib import Path

# COMMAND ----------

# DBTITLE 1,Generating toy data
def create_image(path, img_size, min_radius):
    path.parent.mkdir( parents=True, exist_ok=True )
    
    arr = np.zeros((img_size, img_size)).astype(np.uint8)
    center_x = random.randint(min_radius, (img_size-min_radius))
    center_y = random.randint(min_radius, (img_size-min_radius))
    max_radius = min(center_x, center_y, img_size - center_x, img_size - center_y)
    radius = random.randint(min_radius, max_radius)

    row_indxs, column_idxs = draw.ellipse(center_x, center_y, radius, radius, shape=arr.shape)
    
    arr[row_indxs, column_idxs] = 255

    im = Image.fromarray(arr)
    im.save(path)


def create_images(data_root_path, train_num, val_num, test_num, img_size=640, min_radius=10):
    data_root_path = Path(data_root_path)
    
    for i in range(train_num):
        create_image(data_root_path / 'train' / 'images' / f'img_{i}.png', img_size, min_radius)
        
    for i in range(val_num):
        create_image(data_root_path / 'val' / 'images' / f'img_{i}.png', img_size, min_radius)
        
    for i in range(test_num):
        create_image(data_root_path / 'test' / 'images' / f'img_{i}.png', img_size, min_radius)

create_images('datasets', train_num=120, val_num=40, test_num=40, img_size=120, min_radius=10)

# COMMAND ----------

# DBTITLE 1,Create Data Labels
from rasterio import features

def create_label(image_path, label_path):
    arr = np.asarray(Image.open(image_path))

    # There may be a better way to do it, but this is what I have found so far
    cords = list(features.shapes(arr, mask=(arr >0)))[0][0]['coordinates'][0]
    label_line = '0 ' + ' '.join([f'{int(cord[0])/arr.shape[0]} {int(cord[1])/arr.shape[1]}' for cord in cords])

    label_path.parent.mkdir( parents=True, exist_ok=True )
    with label_path.open('w') as f:
        f.write(label_line)

for images_dir_path in [Path(f'datasets/{x}/images') for x in ['train', 'val', 'test']]:
    for img_path in images_dir_path.iterdir():
        label_path = img_path.parent.parent / 'labels' / f'{img_path.stem}.txt'
        label_line = create_label(img_path, label_path)

# COMMAND ----------

# DBTITLE 1,YAML for Data Config
yaml_content = f'''
train: train/images
val: val/images
test: test/images

names: ['circle']
    '''
    
with Path('data.yaml').open('w') as f:
    f.write(yaml_content)

# COMMAND ----------

# DBTITLE 1,"Tree" structure of the datasets folder and subfolders
# %sh tree .

%ls -lah --recursive datasets/.

# Described in data.yaml 

# COMMAND ----------

# DBTITLE 1,Training
# ref https://github.com/ultralytics/ultralytics/blob/main/ultralytics/engine/trainer.py

# COMMAND ----------

# DBTITLE 1,Ultralytics package/framework | checks
import ultralytics
ultralytics.__version__ #'8.3.34'

ultralytics.checks(verbose=True) 

# ultralytics.YOLO

# COMMAND ----------

# in case existing distributed processes are running we want to stop them before starting a new one
dist.destroy_process_group()

# COMMAND ----------

# DBTITLE 1,Training | Transfer learning
import torch.distributed as dist
from ultralytics import YOLO

# Initialize the process group
dist.init_process_group(backend='nccl')

try:
    model = YOLO("yolov8n-seg.pt")

    results = model.train(
                            batch=8,
                            data="data.yaml",
                            epochs=7, # N runs
                            imgsz=120,
                            # optimizers="adam", #"sgd",
                        )
finally:
    # Destroy the process group
    dist.destroy_process_group()

# COMMAND ----------

# DBTITLE 1,RESULTS
# Results saved to runs/segment/train7

# COMMAND ----------

# DBTITLE 1,Validation Labels
from IPython.display import Image as show_image
# show_image(filename="runs/segment/train60/val_batch0_labels.jpg")
show_image(filename="runs/segment/train7/val_batch0_labels.jpg")

# COMMAND ----------

# DBTITLE 1,Predicted Val Labels
show_image(filename="runs/segment/train7/val_batch0_pred.jpg")

# COMMAND ----------

# DBTITLE 1,Precision | Recall | PR Curves
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Load images
precision_img = mpimg.imread("runs/segment/train7/MaskP_curve.png")
recall_img = mpimg.imread("runs/segment/train7/MaskR_curve.png")
pr_curve_img = mpimg.imread("runs/segment/train7/MaskPR_curve.png")  # Assuming you have a PR curve image

# Create a figure with 3 subplots
fig, axs = plt.subplots(1, 3, figsize=(15, 10))

# Display the images in the subplots
axs[0].imshow(precision_img)
axs[0].set_title('Precision Curve')
axs[0].axis('off')

axs[1].imshow(recall_img)
axs[1].set_title('Recall Curve')
axs[1].axis('off')

axs[2].imshow(pr_curve_img)
axs[2].set_title('PR Curve')
axs[2].axis('off')

# Show the combined figure
plt.tight_layout()
plt.show()

# COMMAND ----------

# DBTITLE 1,Precision
show_image(filename="runs/segment/train7/MaskP_curve.png")

# COMMAND ----------

# DBTITLE 1,Recall
show_image(filename="runs/segment/train7/MaskR_curve.png")

# COMMAND ----------

# DBTITLE 1,PR curve
show_image(filename="runs/segment/train7/MaskPR_curve.png")

# COMMAND ----------

# DBTITLE 1,Training Loss
show_image(filename="runs/segment/train7/results.png")

# COMMAND ----------

# MAGIC %md
# MAGIC

# COMMAND ----------

# DBTITLE 1,Make an example Inference with Trained Model
from ultralytics import YOLO

my_model = YOLO('runs/segment/train7/weights/best.pt')
results = list(my_model('datasets/test/images/img_5.png', conf=0.128))
result = results[0]

# COMMAND ----------

# DBTITLE 1,results list object
results

# COMMAND ----------

# ref: https://docs.ultralytics.com/reference/engine/results/#ultralytics.engine.results.BaseTensor

# COMMAND ----------

# DBTITLE 1,inferenced mask info.
result.masks.xyn

# COMMAND ----------

# DBTITLE 1,inferenced mask data
result.masks.data

# COMMAND ----------

# DBTITLE 1,Show Instance Segmentation prediction mask
import torchvision.transforms as T

T.ToPILImage()(result.masks.data.data[0].squeeze()) 

# COMMAND ----------

# DBTITLE 1,Compare with 'original'
T.ToPILImage()(result.orig_img.squeeze())

# COMMAND ----------



# COMMAND ----------

# DBTITLE 1,Example plots with pyplot instead of T.ToPILImage
import torchvision.transforms as T
import matplotlib.pyplot as plt

# COMMAND ----------

# DBTITLE 1,pyplot predicted mask
# result.masks.data.data is a tensor of shape (N, 1, H, W)

# Select the first mask from the batch and squeeze to remove the channel dimension
mask = result.masks.data.data[0].squeeze()

# Convert the mask to a PIL image
mask_image = T.ToPILImage()(mask)

# Display the mask using matplotlib
plt.imshow(mask_image, cmap='gray')
plt.axis('off')
plt.show()

# COMMAND ----------

# DBTITLE 1,pyplot orig_img
# result.orig_img contains the original image
orig_image = result.orig_img

# Plot the original image
plt.imshow(orig_image)
plt.axis('off')
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC #### For proper model logging with MLflow: 
# MAGIC Please Refer to example notebook [`Computer Vision: YOLO Training Best Practice on Databricks`] for more details for setting up YOLO model with single node+multiple gpus for additional info. wrt 
# MAGIC - `Define Customized PyFunc`
# MAGIC - `SingleNode-MultiGPU version` -- for logging MLflow Custom PyFunc which you define to include the YOLO model + best params 
# MAGIC
# MAGIC ref: https://learn.microsoft.com/en-us/azure/databricks/machine-learning/model-serving/deploy-custom-models
