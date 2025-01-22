# Databricks notebook source
# refs 
# https://learnopencv.com/3d-u-net-brats/#aioseo-dataset-preprocessing
# https://nipy.org/nibabel/nifti_images.html

# test data
# https://www.kaggle.com/datasets/aiocta/brats2023-part-1

# COMMAND ----------

# UC path 
# mmt_mlops_demos.cv.data
# /Volumes/mmt_mlops_demos/cv/data/BraTS2021_00495/

# COMMAND ----------

!pip install nibabel -q
!pip install scikit-learn -q
!pip install tqdm -q
!pip install split-folders -q
!pip install torchinfo -q
!pip install segmentation-models-pytorch-3d -q
!pip install livelossplot -q
!pip install torchmetrics -q
!pip install tensorboard -q

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import os
import random
import splitfolders
from tqdm import tqdm
import nibabel as nib
import glob
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import shutil
import time
 
from dataclasses import dataclass
 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchvision.transforms as transforms
from torch.cuda import amp
 
from torchmetrics import MeanMetric
from torchmetrics.classification import MulticlassAccuracy
 
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary
import gc
 
import segmentation_models_pytorch_3d as smp
 
from livelossplot import PlotLosses
from livelossplot.outputs import MatplotlibPlot, ExtremaPrinter


# COMMAND ----------

# DBTITLE 1,Download test data (BraTS2023) from Kaggle
# https://www.kaggle.com/datasets/aiocta/brats2023-part-1

# Do ONCE
# !pip install kaggle -q
# !kaggle datasets download -d aiocta/brats2023-part-1 -p /Volumes/mmt_mlops_demos/cv/data/BraTS2023/

# COMMAND ----------

# DBTITLE 1,Unzip BraTS2023 within UC Vols
# DO ONCE 
# !sudo apt install unzip
# !unzip /Volumes/mmt_mlops_demos/cv/data/BraTS2023/brats2023-part-1.zip -d /Volumes/mmt_mlops_demos/cv/data/BraTS2023/BraTS2023-Glioma/

# !rm -rf <UD Vols path to>/brats2023-part-1.zip

# COMMAND ----------

# def seed_everything(SEED):
#    np.random.seed(SEED)
#    torch.manual_seed(SEED)
#    torch.cuda.manual_seed_all(SEED)
#    torch.backends.cudnn.deterministic = True
#    torch.backends.cudnn.benchmark = False
 
 
# def get_default_device():
#    gpu_available = torch.cuda.is_available()
#    return torch.device('cuda' if gpu_available else 'cpu'), gpu_available


# COMMAND ----------

# @dataclass(frozen=True)
# class TrainingConfig:
#    BATCH_SIZE:      int = 5
#    EPOCHS:          int = 100
#    LEARNING_RATE: float = 1e-3
#    CHECKPOINT_DIR:  str = os.path.join('model_checkpoint', '3D_UNet_Brats2023')
#    NUM_WORKERS:     int = 4

# COMMAND ----------

scaler = MinMaxScaler()
 
DATASET_PATH = '/Volumes/mmt_mlops_demos/cv/data/BraTS2023/BraTS2023-Glioma/'
print("Total Files: ", len(os.listdir(DATASET_PATH)))
# Total Files:  625

# COMMAND ----------

# Load the NIfTI image
sample_image_flair = nib.load(os.path.join(DATASET_PATH , "BraTS-GLI-00000-000/BraTS-GLI-00000-000-t2f.nii")).get_fdata()
print("Original max value:", sample_image_flair.max()) 
# Original max value: 2934.0
 
# Reshape the 3D image to 2D for scaling
sample_image_flair_flat = sample_image_flair.reshape(-1, 1)

# COMMAND ----------

sample_image_flair

# COMMAND ----------

sample_image_flair_flat

# COMMAND ----------

# Apply scaling
sample_image_flair_scaled = scaler.fit_transform(sample_image_flair_flat)
 
# Reshape it back to the original 3D shape
sample_image_flair_scaled = sample_image_flair_scaled.reshape(sample_image_flair.shape)
 
print("Scaled max value:", sample_image_flair_scaled.max())
print("Shape of scaled Image: ", sample_image_flair_scaled.shape)

# Scaled max value: 1.0
# Shape of scaled Image:  (240, 240, 155)

# COMMAND ----------

sample_mask = nib.load(DATASET_PATH + "BraTS-GLI-00000-000/BraTS-GLI-00000-000-seg.nii").get_fdata()
sample_mask = sample_mask.astype(np.uint8)  # values between 0 and 255
 
print("Unique class in the mask", np.unique(sample_mask)) 
print("Shape of sample_mask: ", sample_mask.shape)

# Unique class in the mask [0 1 2 3]
# Shape of sample_mask:  (240, 240, 155)

# COMMAND ----------

sample_image_t1 = nib.load(DATASET_PATH + "BraTS-GLI-00000-000/BraTS-GLI-00000-000-t1n.nii").get_fdata()
sample_image_t1 = sample_image_t1.astype(np.uint8)  # values between 0 and 255


sample_image_t1c = nib.load(DATASET_PATH + "BraTS-GLI-00000-000/BraTS-GLI-00000-000-t1c.nii").get_fdata()
sample_image_t1c = sample_image_t1c.astype(np.uint8)  # values between 0 and 255


sample_image_t2 = nib.load(DATASET_PATH + "BraTS-GLI-00000-000/BraTS-GLI-00000-000-t2w.nii").get_fdata()
sample_image_t2 = sample_image_t2.astype(np.uint8)  # values between 0 and 255

# COMMAND ----------

import numpy as np

# Define the range
low = 13
high = 90 #141

# Generate a random integer between low (inclusive) and high (exclusive)
rand_int = np.random.randint(low, high)
print(f"Random integer between {low} and {high}: {rand_int}")

# COMMAND ----------

# n_slice = random.randint(0, sample_mask.shape[2])  # random slice between 0 - 154
n_slice = np.random.randint(low, high) #77
print("n_slice: ", n_slice)

plt.figure(figsize = (12,8))
 
plt.subplot(231)
plt.imshow(sample_image_flair_scaled[:,:,n_slice], cmap='gray')
plt.title('Image flair')
 
plt.subplot(232)
plt.imshow(sample_image_t1[:,:,n_slice], cmap = "gray")
plt.title("Image t1")
 
plt.subplot(233)
plt.imshow(sample_image_t1c[:,:,n_slice], cmap='gray')
plt.title("Image t1c")
 
plt.subplot(234)
plt.imshow(sample_image_t2[:,:,n_slice], cmap = 'gray')
plt.title("Image t2")
 
plt.subplot(235)
plt.imshow(sample_mask[:,:,n_slice])
plt.title("Seg Mask")
 
plt.subplot(236)
plt.imshow(sample_mask[:,:,n_slice], cmap = 'gray')
plt.title('Mask Gray')
plt.show()

# COMMAND ----------

combined_x = np.stack(
    [sample_image_flair_scaled, sample_image_t1c, sample_image_t2], axis=3
)  # along the last channel dimension.
print("Shape of Combined x ", combined_x.shape)
# Shape of Combined x  (240, 240, 155, 3)

# COMMAND ----------

combined_x = combined_x[56:184, 56:184, 13:141]
print("Shape after cropping: ", combined_x.shape)
 
sample_mask_c = sample_mask[56:184,56:184, 13:141]
print("Mask shape after cropping: ", sample_mask_c.shape)
 
#Shape after cropping:  (128, 128, 128, 3)
#Mask shape after cropping:  (128, 128, 128)

# COMMAND ----------

plt.figure(figsize = (6,4))

plt.subplot(121)
plt.imshow(combined_x[:,:,n_slice],
          #   cmap = 'gray'
          )
plt.title("combined_x")

plt.subplot(122)
plt.imshow(sample_mask_c[:,:,n_slice], 
          #  cmap = 'gray'
          )
plt.title("sample_mask_c")

# COMMAND ----------

sample_mask_cat  = F.one_hot(torch.tensor(sample_mask_c, dtype = torch.long), num_classes = 4) 

# COMMAND ----------

sample_mask_cat

# COMMAND ----------

# t1ce_list = sorted(glob.glob(f"{DATASET_PATH}/*/*t1c.nii"))
# t2_list = sorted(glob.glob(f"{DATASET_PATH}/*/*t2w.nii"))
# flair_list = sorted(glob.glob(f"{DATASET_PATH}/*/*t2f.nii"))
# mask_list = sorted(glob.glob(f"{DATASET_PATH}/*/*seg.nii"))
 
# print("t1ce list: ", len(t1ce_list))
# print("t2 list: ", len(t2_list))
# print("flair list: ", len(flair_list))
# print("Mask list: ", len(mask_list))

# t1ce list:  625
# t2 list:  625
# flair list:  625
# Mask list:  625

# COMMAND ----------

# to continue with preprocessing for normal pytorch process and then try to convert to coco 
