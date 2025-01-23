# Databricks notebook source
# refs 
# https://learnopencv.com/3d-u-net-brats/#aioseo-dataset-preprocessing (WRT BraTS dataset)

# https://nipy.org/nibabel/nifti_images.html

## NIFTI (brain imaging related but not everyone uses it; DICOM may be preferred)
# https://github.com/DataCurationNetwork/data-primers/blob/main/Neuroimaging%20DICOM%20and%20NIfTI%20Data%20Curation%20Primer/neuroimaging-dicom-and-nifti-data-curation-primer.md
# https://discovery.ucl.ac.uk/id/eprint/10146893/1/geometry_medim.pdf 


# test data
# https://www.kaggle.com/datasets/aiocta/brats2023-part-1

# COMMAND ----------

# UC path 
# mmt_mlops_demos.cv.data
# /Volumes/mmt_mlops_demos/cv/data/BraTS2021_00495/

# COMMAND ----------

## to do -- convert some of the setup as a utils/config file  etc. 

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

# DBTITLE 1,maybe useful for later
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

# DBTITLE 1,Omitting because we are using YOLO instead
# @dataclass(frozen=True)
# class TrainingConfig:
#    BATCH_SIZE:      int = 5
#    EPOCHS:          int = 100
#    LEARNING_RATE: float = 1e-3
#    CHECKPOINT_DIR:  str = os.path.join('model_checkpoint', '3D_UNet_Brats2023')
#    NUM_WORKERS:     int = 4

# COMMAND ----------

# DBTITLE 1,scaler = MinMaxScaler()
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

# DBTITLE 1,probably want to apply scaling  to the other image types?
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
# sample_image_t1 = sample_image_t1.astype(np.uint8)  # values between 0 and 255 | NOT NEEDED?


sample_image_t1ce = nib.load(DATASET_PATH + "BraTS-GLI-00000-000/BraTS-GLI-00000-000-t1c.nii").get_fdata()
# sample_image_t1c = sample_image_t1c.astype(np.uint8)  # values between 0 and 255 | NOT NEEDED?


sample_image_t2 = nib.load(DATASET_PATH + "BraTS-GLI-00000-000/BraTS-GLI-00000-000-t2w.nii").get_fdata()
# sample_image_t2 = sample_image_t2.astype(np.uint8)  # values between 0 and 255 |  NOT NEEDED?

# COMMAND ----------

# DBTITLE 1,test randint
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
plt.imshow(sample_image_t1ce[:,:,n_slice], cmap='gray')
plt.title("Image t1ce")
 
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
    [sample_image_flair_scaled, sample_image_t1ce, sample_image_t2], axis=3
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

# DBTITLE 1,Curious what this 'looks' like hmm
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

t1ce_list = sorted(glob.glob(f"{DATASET_PATH}/*/*t1c.nii"))
t2_list = sorted(glob.glob(f"{DATASET_PATH}/*/*t2w.nii"))
flair_list = sorted(glob.glob(f"{DATASET_PATH}/*/*t2f.nii"))
mask_list = sorted(glob.glob(f"{DATASET_PATH}/*/*seg.nii"))
 
print("t1ce list: ", len(t1ce_list))
print("t2 list: ", len(t2_list))
print("flair list: ", len(flair_list))
print("Mask list: ", len(mask_list))

# t1ce list:  625
# t2 list:  625
# flair list:  625
# Mask list:  625

# COMMAND ----------

# to continue with preprocessing for normal pytorch process and then try to convert to coco 

# COMMAND ----------

## DATASET Preprocessing test to pytorch dataloader 
# -- we will need to see how to reformat to coco/yolo friendly format 

# COMMAND ----------

# '/'.join(f"{DATASET_PATH}".split("/")[:-2])
UCV_folderpath =  "/Volumes/mmt_mlops_demos/cv/data/BraTS2023/"

# COMMAND ----------

# DBTITLE 1,maybe there's a way to distribute this via spark udf OR streaming?
for idx in tqdm(
    range(len(t2_list)), desc="Preparing to stack, crop and save", unit="file"
):
    temp_image_t1ce = nib.load(t1ce_list[idx]).get_fdata()
    temp_image_t1ce = scaler.fit_transform(
        temp_image_t1ce.reshape(-1, temp_image_t1ce.shape[-1])
    ).reshape(temp_image_t1ce.shape)
 
    temp_image_t2 = nib.load(t2_list[idx]).get_fdata()
    temp_image_t2 = scaler.fit_transform(
        temp_image_t2.reshape(-1, temp_image_t2.shape[-1])
    ).reshape(temp_image_t2.shape)
 
    temp_image_flair = nib.load(flair_list[idx]).get_fdata()
    temp_image_flair = scaler.fit_transform(
        temp_image_flair.reshape(-1, temp_image_flair.shape[-1])
    ).reshape(temp_image_flair.shape)
 
    temp_mask = nib.load(mask_list[idx]).get_fdata()
 
    temp_combined_images = np.stack(
        [temp_image_flair, temp_image_t1ce, temp_image_t2], axis=3
    )
 
    temp_combined_images = temp_combined_images[56:184, 56:184, 13:141]
    temp_mask = temp_mask[56:184, 56:184, 13:141]
 
    val, counts = np.unique(temp_mask, return_counts=True)
 
    # If a volume has less than 1% of mask, we simply ignore to reduce computation
    if (1 - (counts[0] / counts.sum())) > 0.01:
        #         print("Saving Processed Images and Masks")
        temp_mask = F.one_hot(torch.tensor(temp_mask, dtype=torch.long), num_classes=4)
        os.makedirs(f"{UCV_folderpath}BraTS2023_Preprocessed/input_data_3channels/images", exist_ok=True)
        os.makedirs(f"{UCV_folderpath}BraTS2023_Preprocessed/input_data_3channels/masks", 
                    exist_ok=True)
 
        np.save(
            f"{UCV_folderpath}BraTS2023_Preprocessed/input_data_3channels/images/image_"
            + str(idx)
            + ".npy",
            temp_combined_images,
        )
        np.save(
            f"{UCV_folderpath}BraTS2023_Preprocessed/input_data_3channels/masks/mask_"
            + str(idx)
            + ".npy",
            temp_mask,
        )
 
    else:
        pass

# COMMAND ----------

images_folder = f"{UCV_folderpath}BraTS2023_Preprocessed/input_data_3channels/images"
print(len(os.listdir(images_folder)))
 
masks_folder = f"{UCV_folderpath}BraTS2023_Preprocessed/input_data_3channels/masks"
print(len(os.listdir(masks_folder)))

# Images: 575
# Masks: 575

# COMMAND ----------

# DBTITLE 1,split into train / val
input_folder = f"{UCV_folderpath}BraTS2023_Preprocessed/input_data_3channels/"
 
output_folder = f"{UCV_folderpath}BraTS2023_Preprocessed/input_data_128/"
 
splitfolders.ratio(
    input_folder, output_folder, seed=42, ratio=(0.75, 0.25), group_prefix=None
)

# COMMAND ----------

# DBTITLE 1,NOT deleting files
# if os.path.exists(input_folder):
#     shutil.rmtree(input_folder)
#     print(f"{input_folder} is removed")
# else:
#     print(f"{input_folder} doesn't exist")

# COMMAND ----------

# DBTITLE 1,define a custom data loader?
class BraTSDataset(Dataset):
    def __init__(self, img_dir, mask_dir, normalization=True):
        super().__init__()
 
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_list = sorted(
            os.listdir(img_dir)
        )  # Ensure sorting to match images and masks
        self.mask_list = sorted(os.listdir(mask_dir))
        self.normalization = normalization
 
        # If normalization is True, set up a normalization transform
        if self.normalization:
            self.normalizer = transforms.Normalize(
                mean=[0.5], std=[0.5]
            )  # Adjust mean and std based on your data
 
    def load_file(self, filepath):
        return np.load(filepath)
 
    def __len__(self):
        return len(self.img_list)
    
    def __getitem__(self, idx):
       image_path = os.path.join(self.img_dir, self.img_list[idx])
       mask_path = os.path.join(self.mask_dir, self.mask_list[idx])
       # Load the image and mask
       image = self.load_file(image_path)
       mask = self.load_file(mask_path)
 
       # Convert to torch tensors and permute axes to C, D, H, W format (needed for 3D models)
       image = torch.from_numpy(image).permute(3, 2, 0, 1)  # Shape: C, D, H, W
       mask = torch.from_numpy(mask).permute(3, 2, 0, 1)  # Shape: C, D, H, W
       
       # Normalize the image if normalization is enabled
       if self.normalization:
           image = self.normalizer(image)
       
       return image, mask

# COMMAND ----------

train_img_dir = f"{UCV_folderpath}BraTS2023_Preprocessed/input_data_128/train/images"
train_mask_dir = f"{UCV_folderpath}BraTS2023_Preprocessed/input_data_128/train/masks"
 
val_img_dir = f"{UCV_folderpath}BraTS2023_Preprocessed/input_data_128/val/images"
val_mask_dir = f"{UCV_folderpath}BraTS2023_Preprocessed/input_data_128/val/masks"
 
val_img_list = os.listdir(val_img_dir)
val_mask_list = os.listdir(val_mask_dir)
 
# Initialize datasets with normalization only
train_dataset = BraTSDataset(train_img_dir, train_mask_dir, normalization=True)
val_dataset = BraTSDataset(val_img_dir, val_mask_dir, normalization=True)
 
# Print dataset statistics
print("Total Training Samples: ", len(train_dataset))
print("Total Val Samples: ", len(val_dataset))

#Total Training Samples:  431
#Total Val Samples:  144

# COMMAND ----------

train_loader = DataLoader(train_dataset, batch_size=5, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=5, shuffle=False, num_workers=4)

# Sanity Check
images, masks = next(iter(train_loader))
print(f"Train Image batch shape: {images.shape}")
print(f"Train Mask batch shape: {masks.shape}")

train_loader = DataLoader(train_dataset, batch_size=5, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=5, shuffle=False, num_workers=4)

# Sanity Check
images, masks = next(iter(train_loader))
print(f"Train Image batch shape: {images.shape}")
print(f"Train Mask batch shape: {masks.shape}")

# Train Image batch shape: torch.Size([5, 3, 128, 128, 128])
# Train Mask batch shape: torch.Size([5, 4, 128, 128, 128])

# COMMAND ----------

def visualize_slices(images, masks, num_slices=20):
    batch_size = images.shape[0]
 
    masks = torch.argmax(masks, dim=1)  # along the channel/class dim
 
    for i in range(min(num_slices, batch_size)):
        fig, ax = plt.subplots(1, 5, figsize=(15, 5))
 
        middle_slice = images.shape[2] // 2
        ax[0].imshow(images[i, 0, middle_slice, :, :], cmap="gray")
        ax[1].imshow(images[i, 1, middle_slice, :, :], cmap="gray")
        ax[2].imshow(images[i, 2, middle_slice, :, :], cmap="gray")
        ax[3].imshow(masks[i, middle_slice, :, :], cmap="viridis")
        ax[4].imshow(masks[i, middle_slice, :, :], cmap="gray")
 
        ax[0].set_title("T1ce")
        ax[1].set_title("FLAIR")
        ax[2].set_title("T2")
        ax[3].set_title("Seg Mask")
        ax[4].set_title("Mask - Gray")
 
        plt.show()
 
 
visualize_slices(images, masks, num_slices=20)

# COMMAND ----------


