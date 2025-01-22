# Databricks notebook source
# MAGIC %md
# MAGIC ## Computer Vision: CNN and YOLO

# COMMAND ----------

# MAGIC %md
# MAGIC ref: 
# MAGIC 1. on databricks, https://benhay.es/posts/object_detection_yolov8/
# MAGIC 2. generic, https://towardsdatascience.com/trian-yolov8-instance-segmentation-on-your-data-6ffa04b2debd
# MAGIC
# MAGIC ![](https://raw.githubusercontent.com/ultralytics/assets/main/im/banner-tasks.png)
# MAGIC

# COMMAND ----------

# %pip install ultralytics==8.1.14 opencv-python==4.8.0.74

# COMMAND ----------

# MAGIC %pip install -U ultralytics==8.3.31 opencv-python==4.10.0.84 

# COMMAND ----------

# %pip install ultralytics==8.2.2

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import ultralytics
print(ultralytics.__version__)

# COMMAND ----------

from ultralytics.utils.checks import check_yolo, check_python, check_latest_pypi_version, check_version, check_requirements

print("check_yolo", check_yolo())
print("check_python", check_python())
print("check_latest_pypi_version", check_latest_pypi_version())
print("check_version", check_version())

# COMMAND ----------

# ultralytics.checks(verbose=True, device=[0,1])

# COMMAND ----------

import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1" # CUDA_LAUNCH_BLOCKING make cuda report the error where it actually occurs.

# COMMAND ----------

from ultralytics import YOLO
import torch
import mlflow
import torch.distributed as dist
from ultralytics import settings
from mlflow.types.schema import Schema, ColSpec
from mlflow.models.signature import ModelSignature

input_schema = Schema(
    [
        ColSpec("string", "image_source"),
    ]
)
output_schema = Schema([ColSpec("string","class_name"),
                        ColSpec("integer","class_num"),
                        ColSpec("double","confidence")]
                       )

signature = ModelSignature(inputs=input_schema, 
                           outputs=output_schema)

settings.update({"mlflow":False})

# COMMAND ----------

# os.environ["OMP_NUM_THREADS"] = "12"  # OpenMP threads

# COMMAND ----------

############################################################################
## Create YOLOC class to capture model results d a predict() method ##
############################################################################

class YOLOC(mlflow.pyfunc.PythonModel):
  def __init__(self, point_file):
    self.point_file=point_file

  def load_context(self, context):
    from ultralytics import YOLO
    self.model = YOLO(context.artifacts['best_point'])

# COMMAND ----------

# MAGIC %sql
# MAGIC create catalog if not exists yyang;
# MAGIC create schema if not exists yyang.computer_vision;
# MAGIC create Volume if not exists yyang.computer_vision.yolo;

# COMMAND ----------

# Config MLflow
mlflow.autolog(disable=False)
mlflow.end_run()

# Config project structure directory
project_location = '/Volumes/yyang/computer_vision/yolo/'
os.makedirs(f'{project_location}/training_runs/', exist_ok=True)
os.chdir(f'{project_location}/training_runs/')

os.makedirs(f'{project_location}/training_results/', exist_ok=True)
os.chdir(f'{project_location}/training_results/')

# COMMAND ----------

if not dist.is_initialized():
  # import torch.distributed as dist
  dist.init_process_group("nccl")

# COMMAND ----------

mlflow.end_run()
with mlflow.start_run():
  # model = YOLO("yolov8l-seg.pt")
  model = YOLO("yolov8n.pt")

  model.train(
          batch=8,
          device=[0,1],
          # data=f"{project_location}/training/data.yaml",
          data=f"coco8.yaml",
          epochs=50,
          # project='/tmp/solar_panel_damage/',
          project=f'{project_location}/training_results/',
          exist_ok=True,
          fliplr=1,
          flipud=1,
          perspective=0.001,
          degrees=.45
      )

  mlflow.log_params(vars(model.trainer.model.args))
  yolo_wrapper = YOLOC(model.trainer.best)
  mlflow.pyfunc.log_model(artifact_path = "model",
                          artifacts = {'model_path': str(model.trainer.save_dir), 
                                       "best_point": str(model.trainer.best)},
                          python_model = yolo_wrapper,
                          signature = signature
                          )

# COMMAND ----------

mlflow.end_run()

from pyspark.ml.torch.distributor import TorchDistributor

def train_fn():

  from ultralytics import YOLO
  import torch
  import mlflow
  import torch.distributed as dist
  from ultralytics import settings
  from mlflow.types.schema import Schema, ColSpec
  from mlflow.models.signature import ModelSignature

  if not dist.is_initialized():
    # import torch.distributed as dist
    dist.init_process_group("nccl")

  model = YOLO("yolov8n.pt")
  model.train(
      batch=8,
      device=[0, 1],
      data="coco8.yaml",
      epochs=50,
      project=f'{project_location}/training_results/',
      exist_ok=True,
      fliplr=1,
      flipud=1,
      perspective=0.001,
      degrees=.45
  )
  mlflow.log_params(vars(model.trainer.model.args))
  yolo_wrapper = YOLOC(model.trainer.best)
  mlflow.pyfunc.log_model(
      artifact_path="model",
      artifacts={'model_path': str(model.trainer.save_dir), "best_point": str(model.trainer.best)},
      python_model=yolo_wrapper,
      signature=signature
  )

distributor = TorchDistributor(num_processes=2, local_mode=True, use_gpu=True )
distributor.run(train_fn)

# COMMAND ----------

import torch

num_cuda_devices = torch.cuda.device_count()
num_cuda_devices

# COMMAND ----------

import os
del os.environ['LOCAL_RANK']

# COMMAND ----------

from ultralytics import settings
settings.update({'mlflow': False})

# COMMAND ----------

# import torch; torch.distributed.init_process_group(backend='nccl')

# COMMAND ----------

# from ultralytics import YOLO
# import os

# # os.environ["NCCL_P2P_DISABLE"] = "1" # not sure if this is needed, at least this was only flagged as useful for AWS g5 instances

# os.environ["NCCL_P2P_DISABLE"] = "0"

# model = YOLO('yolov8n.pt')
# results = model.train(data='coco128.yaml', epochs=8, device=[0,1])

# COMMAND ----------

# import torch
# import os
# from ultralytics import YOLO

# # # Disable CUDA initialization
# # os.environ["CUDA_VISIBLE_DEVICES"] = ""


# # # Ensure CUDA is not initialized before using fork
# # if torch.cuda.is_initialized():
# #     raise Exception("CUDA was initialized; distributed training will fail with fork start method.")

# def train_fn(rank, epochs, data):
#     # Setup for distributed training, e.g., setting up environment variables for each process
#     # Your training code here, for example:
#     model = YOLO('yolov8n.pt')
#     results = model.train(data=data, epochs=epochs, device=rank)

# # Number of processes to start
# num_processes = 2
# # Arguments to pass to the training function
# args = (8, 'coco128.yaml')  # epochs, data

# # Start processes
# torch.multiprocessing.start_processes(
#     train_fn, 
#     args=args, 
#     nprocs=num_processes, 
#     start_method="fork"
# )

# COMMAND ----------

from ultralytics import settings
settings.update({'mlflow': True})

# COMMAND ----------

model.tune(data='coco8.yaml', epochs=8, iterations=4, optimizer='AdamW', plots=False, save=False, val=True, device=[0,1,2,3])

# COMMAND ----------

# Save the fine-tuned model
model.save('yolov8n_finetuned.pt')
