# Databricks notebook source
# MAGIC %md
# MAGIC ## Computer Vision: CNN and YOLO

# COMMAND ----------

# MAGIC %md
# MAGIC ref: 
# MAGIC 1. YOLO on databricks, https://benhay.es/posts/object_detection_yolov8/
# MAGIC 2. Distributed Torch + MLflow, https://docs.databricks.com/en/machine-learning/train-model/distributed-training/spark-pytorch-distributor.html
# MAGIC
# MAGIC ![](https://raw.githubusercontent.com/ultralytics/assets/main/im/banner-tasks.png)
# MAGIC

# COMMAND ----------

# DBTITLE 1,asIs env from ref
# %pip install ultralytics==8.1.14 opencv-python==4.8.0.74

# import os
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


# COMMAND ----------

# DBTITLE 1,newest env as of today
# MAGIC %pip install -U ultralytics==8.3.31 opencv-python==4.10.0.84 ray==2.39.0

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,debug switcher
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1" # Enable synchronous execution for debugging,for debugging purpose, show stack trace immediately. Use it in dev mode.
# os.environ["CUDA_LAUNCH_BLOCKING"] = "0" # Reset to asynchronous execution for production in production

# COMMAND ----------

# DBTITLE 1,Super Important for MLflow
# Set Databricks workspace URL and token
os.environ['DATABRICKS_HOST'] = db_host = 'https://adb-984752964297111.11.azuredatabricks.net'
os.environ['DATABRICKS_WORKSPACE_ID'] = db_wksp_id = '984752964297111'
os.environ['DATABRICKS_TOKEN'] = db_token = 'dapi7d5fbb5971299484fdf3322d51b84e62'

# COMMAND ----------

import ultralytics
print(ultralytics.__version__)

# COMMAND ----------

import ray
ray.__version__

# COMMAND ----------

from ultralytics.utils.checks import check_yolo, check_python, check_latest_pypi_version, check_version, check_requirements

print("check_yolo", check_yolo())
print("check_python", check_python())
print("check_latest_pypi_version", check_latest_pypi_version())
print("check_version", check_version())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Define Customized PyFunc

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

# settings.update({"mlflow":False}) # Specifically, it disables the integration with MLflow. By setting the mlflow key to False, you are instructing the ultralytics library not to use MLflow for logging or tracking experiments.

# ultralytics level setting with MLflow
settings.update({"mlflow":True}) # if you do want to autolog.
# # Config MLflow
mlflow.autolog(disable=True)
mlflow.end_run()

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

# MAGIC %md
# MAGIC ## Setup I/O Path

# COMMAND ----------

# MAGIC %sql
# MAGIC create catalog if not exists yyang;
# MAGIC create schema if not exists yyang.computer_vision;
# MAGIC create Volume if not exists yyang.computer_vision.yolo;

# COMMAND ----------

import os
# Config project structure directory
project_location = '/Volumes/yyang/computer_vision/yolo/'
os.makedirs(f'{project_location}/training_runs/', exist_ok=True)
# os.chdir(f'{project_location}/training_runs/')
os.makedirs(f'{project_location}/data/', exist_ok=True)
os.makedirs(f'{project_location}/raw_model/', exist_ok=True)

# for cache related to ultralytics
os.environ['ULTRALYTICS_CACHE_DIR'] = f'{project_location}/raw_model/'


# volume folder in UC.
volume_project_location = f'{project_location}/training_results/'
os.makedirs(volume_project_location, exist_ok=True)

# or more traditional way, setup folder under DBFS.
# dbfs_project_location = '/dbfs/FileStore/cv_project_location/yolo/'
dbfs_project_location = '/dbfs/tmp/cv_project_location/yolo/'
os.makedirs(dbfs_project_location, exist_ok=True)

# ephemeral /tmp/ project location on VM
tmp_project_location = "/tmp/training_results/"
os.makedirs(tmp_project_location, exist_ok=True)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Download Dataset using .yaml template

# COMMAND ----------

dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get().rsplit('/', 1)[0]

# COMMAND ----------

os.chdir('/Workspace/' + dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get().rsplit('/', 1)[0])
os.getcwd()

# COMMAND ----------

# MAGIC %sh
# MAGIC curl -L https://github.com/ultralytics/ultralytics/raw/main/ultralytics/cfg/datasets/coco8.yaml -o coco8.yaml

# COMMAND ----------

import os

with open('coco8.yaml', 'r') as file:
    data = file.read()

data = data.replace('../datasets/', f'{project_location}/data/')

with open('coco8.yaml', 'w') as file:
    file.write(data)

# COMMAND ----------

# MAGIC %sh
# MAGIC cat ./coco8.yaml

# COMMAND ----------

import yaml

with open('coco8.yaml', 'r') as file:
    data = yaml.safe_load(file)

data

# COMMAND ----------

# DBTITLE 1,Extract Zip File from URL and Save to Path
import requests, zipfile, io

response = requests.get(data['download'])
z = zipfile.ZipFile(io.BytesIO(response.content))
z.extractall(data['path'])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Start MLflow Training

# COMMAND ----------

# DBTITLE 1,A simple multi-gpu coop test
import os
import torch
import torch.distributed as dist
from pyspark.ml.torch.distributor import TorchDistributor


def main():
    # Initialize distributed training
    dist.init_process_group(backend='nccl')  # Or 'gloo' if not using GPUs
    rank = dist.get_rank()

    output_file_path = 'output.txt'

    # Only rank 0 should check and create the file
    if rank == 0:
        if not os.path.exists(output_file_path):
            with open(output_file_path, 'w') as f:
                f.write('Hello from rank 0!\n')

    # Barrier to ensure all processes wait until rank 0 creates the file
    dist.barrier()

    # Other ranks can now access the file
    with open(output_file_path, 'a') as f:
        f.write(f'Hello from rank {rank}!\n')

if __name__ == '__main__':
    # main()
    distributor = TorchDistributor(num_processes=4, local_mode=True, use_gpu=True)
    distributor.run(main)

# COMMAND ----------

# MAGIC %cat output.txt

# COMMAND ----------

import os
os.environ["TORCHELASTIC_ERROR_FILE"] = "/tmp/torchelastic_error.json"


# COMMAND ----------

with open("/tmp/torchelastic_error.json", "r") as f:
       error_details = f.read()
       print(error_details)

# COMMAND ----------

mlflow.autolog(disable = True) # this flag does nothing as being dominated by the above setting: settings.update({"mlflow":False}).

# COMMAND ----------

# %pip install databricks-sdk==0.37.0 --upgrade

# COMMAND ----------

# DBTITLE 1,Version A, works
import mlflow
from pyspark.ml.torch.distributor import TorchDistributor


def train_fn():
    from ultralytics import YOLO
    import torch.distributed as dist

    # # Ensure the distributed process group is initialized only once
    # if not dist.is_initialized():
    #     dist.init_process_group(backend='nccl')

    # Initialize the YOLO model
    model = YOLO("yolov8n.pt")

    # Train the model
    model.train(
        batch=8,
        device=[0, 1],
        data="./coco8.yaml",  # Ensure this path is valid and writable
        epochs=50,
        # project=volume_project_location, # in multi-GPU mode, still fails due to 2nd write into csv wont append. OSError: [Errno 95] Operation not supported
        project=tmp_project_location, # this works without mlflow logging
        exist_ok=True,
        fliplr=1,
        flipud=1,
        perspective=0.001,
        degrees=0.45
    )

    # # Log parameters and model using MLflow
    # mlflow.log_params(vars(model.trainer.model.args))
    # yolo_wrapper = YOLOC(model.trainer.best)
    # mlflow.pyfunc.log_model(
    #     artifact_path="model",
    #     artifacts={'model_path': str(model.trainer.save_dir), "best_point": str(model.trainer.best)},
    #     python_model=yolo_wrapper,
    #     signature=signature
    # )

    # # Clean up the process group
    # dist.destroy_process_group()

distributor = TorchDistributor(num_processes=2, local_mode=True, use_gpu=True)
distributor.run(train_fn)

# COMMAND ----------

# DBTITLE 1,Version B, stuck forever
#: (DONT RUN)
from ultralytics import YOLO
import torch.distributed as dist

# # Ensure the distributed process group is initialized only once
# if not dist.is_initialized():
#     dist.init_process_group(backend='nccl')

# Initialize the YOLO model
model = YOLO("yolov8n.pt")

# Train the model
model.train(
    batch=8,
    device=[0, 1],
    data="./coco8.yaml",  # Ensure this path is valid and writable
    epochs=50,
    # project=volume_project_location, # in multi-GPU mode, still fails
    project=tmp_project_location, # this works without mlflow logging
    exist_ok=True,
    fliplr=1,
    flipud=1,
    perspective=0.001,
    degrees=0.45
)

# COMMAND ----------

# DBTITLE 1,version C, fails due to experiement irregular name?
import mlflow
from pyspark.ml.torch.distributor import TorchDistributor

def train_fn():
    from ultralytics import YOLO
    import torch.distributed as dist

    # # Ensure the distributed process group is initialized only once
    # if not dist.is_initialized():
    #     dist.init_process_group("nccl")

    # Initialize the YOLO model
    model = YOLO("yolov8n.pt")

    # Train the model
    model.train(
        batch=8,
        device=[0, 1],
        data="./coco8.yaml",  # Ensure this path is valid and writable
        epochs=50,
        project=volume_project_location,
        exist_ok=True,
        fliplr=1,
        flipud=1,
        perspective=0.001,
        degrees=0.45
    )

    # Log parameters and model using MLflow
    mlflow.log_params(vars(model.trainer.model.args))
    yolo_wrapper = YOLOC(model.trainer.best)
    mlflow.pyfunc.log_model(
        artifact_path="model",
        artifacts={'model_path': str(model.trainer.save_dir), "best_point": str(model.trainer.best)},
        python_model=yolo_wrapper,
        signature=signature
    )

    # # Clean up the process group
    # dist.destroy_process_group()

distributor = TorchDistributor(num_processes=2, local_mode=True, use_gpu=True)
distributor.run(train_fn)

# COMMAND ----------

# DBTITLE 1,version D, stuck for ever
mlflow.end_run()

mlflow.set_experiment(f"/Workspace/Users/yang.yang@databricks.com/ConvaTec Workshop Computer Vision/Experiments_YOLO_CoCo")
with mlflow.start_run():
  # model = YOLO("yolov8l-seg.pt")
  model = YOLO("yolov8n.pt")

  # #: NCCL connection is using server socket
  # if dist.is_initialized():
  #   dist.destroy_process_group() # make sure we dont double init_process_group, so destroy it first if any

  # if not dist.is_initialized():
  #   # import torch.distributed as dist
  #   dist.init_process_group("nccl")

  model.train(
          batch=8,
          device=[0,1],
          # data=f"{project_location}/training/data.yaml",
          data=f"./coco8.yaml",
          epochs=50,
          # project='/tmp/solar_panel_damage/',
          # project=dbfs_project_location, # this will fail on the 2nd attempt to write into csv in appending mode
          project=volume_project_location, # this will fail on the 2nd attempt to write into csv in appending mode
        #   project=tmp_project_location, # this will work as folder is ephemeral and without RBAC.
          exist_ok=True,
          fliplr=1,
          flipud=1,
          perspective=0.001,
          degrees=.45
      )

  mlflow.log_params(vars(model.trainer.model.args))
  yolo_wrapper = YOLOC(model.trainer.best) # creates an instance of YOLOC with the best model checkpoint.
  mlflow.pyfunc.log_model(artifact_path = "model",
                          artifacts = {'model_path': str(model.trainer.save_dir), 
                                       "best_point": str(model.trainer.best)},
                          python_model = yolo_wrapper,
                          signature = signature
                          )

# COMMAND ----------

# DBTITLE 1,version E, stuck for ever
mlflow.end_run()

# Set the existing experiment
mlflow.set_experiment(f"/Users/yang.yang@databricks.com/ConvaTec Workshop Computer Vision/Experiments_YOLO_CoCo")
with mlflow.start_run():
  # model = YOLO("yolov8l-seg.pt")
  model = YOLO("yolov8n.pt")

  # #: NCCL connection is using server socket
  # if dist.is_initialized():
  #   dist.destroy_process_group() # make sure we dont double init_process_group, so destroy it first if any

  # if not dist.is_initialized():
  #   # import torch.distributed as dist
  #   dist.init_process_group("nccl")

  model.train(
          batch=8,
          device=[0,1],
          # data=f"{project_location}/training/data.yaml",
          data=f"./coco8.yaml",
          epochs=50,
          # project='/tmp/solar_panel_damage/',
          # project=dbfs_project_location, # this will fail on the 2nd attempt to write into csv in appending mode
          # project=volume_project_location, # this will fail on the 2nd attempt to write into csv in appending mode
          project=tmp_project_location, # this will work as folder is ephemeral and without RBAC.
          exist_ok=True,
          fliplr=1,
          flipud=1,
          perspective=0.001,
          degrees=.45
      )

  mlflow.log_params(vars(model.trainer.model.args))
  yolo_wrapper = YOLOC(model.trainer.best) # creates an instance of YOLOC with the best model checkpoint.
  mlflow.pyfunc.log_model(artifact_path = "model",
                          artifacts = {'model_path': str(model.trainer.save_dir), 
                                       "best_point": str(model.trainer.best)},
                          python_model = yolo_wrapper,
                          signature = signature
                          )

# COMMAND ----------

# DBTITLE 1,version F, fail, experiment name issue due to project name

from pyspark.ml.torch.distributor import TorchDistributor
mlflow.end_run()
mlflow.set_experiment(f"/Users/yang.yang@databricks.com/ConvaTec Workshop Computer Vision/Experiments_YOLO_CoCo")

def train_fn():

  from ultralytics import YOLO
  import torch
  import mlflow
  import torch.distributed as dist
  from ultralytics import settings
  from mlflow.types.schema import Schema, ColSpec
  from mlflow.models.signature import ModelSignature

  # if not dist.is_initialized():
  #   # import torch.distributed as dist
  #   dist.init_process_group("nccl")

  model = YOLO(f"{volume_project_location}/yolov8n.pt")
  model.train(
      batch=8,
      device=[0, 1],
      data="./coco8.yaml", # ref: https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco8.yaml
      epochs=50,
      project=f'{volume_project_location}',
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

with mlflow.start_run() as run:  
  distributor = TorchDistributor(num_processes=2, local_mode=True, use_gpu=True )
  distributor.run(train_fn)

# COMMAND ----------

# DBTITLE 1,version G, fn arg no attribute failure

from pyspark.ml.torch.distributor import TorchDistributor
mlflow.end_run()
mlflow.set_experiment(f"/Users/yang.yang@databricks.com/ConvaTec Workshop Computer Vision/Experiments_YOLO_CoCo")

def train_fn():

  from ultralytics import YOLO
  import torch
  import mlflow
  import torch.distributed as dist
  from ultralytics import settings
  from mlflow.types.schema import Schema, ColSpec
  from mlflow.models.signature import ModelSignature

  # if not dist.is_initialized():
  #   # import torch.distributed as dist
  #   dist.init_process_group("nccl")

  model = YOLO(f"{volume_project_location}/yolov8n.pt")
  model.train(
      batch=8,
      device=[0, 1],
      data="./coco8.yaml", # ref: https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco8.yaml
      epochs=50,
      project=f'{tmp_project_location}',
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

with mlflow.start_run() as run:  
  distributor = TorchDistributor(num_processes=2, local_mode=True, use_gpu=True )
  distributor.run(train_fn)


#  AttributeError: 'DistributedDataParallel' object has no attribute 'args'

# COMMAND ----------

# DBTITLE 1,version H, failure due to missing experiment name

from pyspark.ml.torch.distributor import TorchDistributor
mlflow.end_run()
mlflow.set_experiment(f"/Users/yang.yang@databricks.com/ConvaTec Workshop Computer Vision/Experiments_YOLO_CoCo")

def train_fn():

  from ultralytics import YOLO
  import torch
  import mlflow
  import torch.distributed as dist
  from ultralytics import settings
  from mlflow.types.schema import Schema, ColSpec
  from mlflow.models.signature import ModelSignature

  # if not dist.is_initialized():
  #   # import torch.distributed as dist
  #   dist.init_process_group("nccl")

  model = YOLO(f"{volume_project_location}/yolov8n.pt")
  model.train(
      batch=8,
      device=[0, 1],
      data="./coco8.yaml", # ref: https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco8.yaml
      epochs=50,
      project=f'{tmp_project_location}',
      exist_ok=True,
      fliplr=1,
      flipud=1,
      perspective=0.001,
      degrees=.45
  )
  # mlflow.log_params(vars(model.trainer.model.args))
  yolo_wrapper = YOLOC(model.trainer.best)
  mlflow.pyfunc.log_model(
      artifact_path="model",
      artifacts={'model_path': str(model.trainer.save_dir), "best_point": str(model.trainer.best)},
      python_model=yolo_wrapper,
      signature=signature
  )

with mlflow.start_run() as run:  
  distributor = TorchDistributor(num_processes=2, local_mode=True, use_gpu=True )
  distributor.run(train_fn)

#  mlflow.exceptions.RestException: RESOURCE_DOES_NOT_EXIST: No experiment was found. If using the Python fluent API, you can set an active experiment under which to create runs by calling mlflow.set_experiment("experiment_name") at the start of your program.

# COMMAND ----------

# DBTITLE 1,version I, AI solution doesn't solve
from pyspark.ml.torch.distributor import TorchDistributor
import mlflow
import os

os.environ["NCCL_SOCKET_IFNAME"] = "eth0"  # Set the network interface for NCCL

mlflow.end_run()
mlflow.set_experiment(f"/Users/yang.yang@databricks.com/ConvaTec Workshop Computer Vision/Experiments_YOLO_CoCo")

def train_fn():
    from ultralytics import YOLO
    import torch
    import mlflow
    import torch.distributed as dist
    from ultralytics import settings
    from mlflow.types.schema import Schema, ColSpec
    from mlflow.models.signature import ModelSignature

    model = YOLO("yolov8n.pt")
    model.train(
        batch=8,
        device=[0, 1],
        data="./coco8.yaml",  # ref: https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco8.yaml
        epochs=50,
        project=tmp_project_location,  # Use a valid project location
        exist_ok=True,
        fliplr=1,
        flipud=1,
        perspective=0.001,
        degrees=0.45
    )

    yolo_wrapper = YOLOC(model.trainer.best)
    mlflow.pyfunc.log_model(
        artifact_path="model",
        artifacts={'model_path': str(model.trainer.save_dir), "best_point": str(model.trainer.best)},
        python_model=yolo_wrapper,
        signature=signature
    )

with mlflow.start_run() as run:
    distributor = TorchDistributor(num_processes=2, local_mode=True, use_gpu=True)
    distributor.run(train_fn)


# rank1]: mlflow.exceptions.RestException: RESOURCE_DOES_NOT_EXIST: No experiment was found. If using the Python fluent API, you can set an active experiment under which to create runs by calling mlflow.set_experiment("experiment_name") at the start of your program.

# COMMAND ----------

# DBTITLE 1,version J, pass, but 3 runs yield
from pyspark.ml.torch.distributor import TorchDistributor
mlflow.end_run()
# mlflow.set_experiment(f"/Users/yang.yang@databricks.com/ConvaTec Workshop Computer Vision/Experiments_YOLO_CoCo")

def train_fn():

  from ultralytics import YOLO
  import torch
  import mlflow
  import torch.distributed as dist
  from ultralytics import settings
  from mlflow.types.schema import Schema, ColSpec
  from mlflow.models.signature import ModelSignature

  # if not dist.is_initialized():
  #   # import torch.distributed as dist
  #   dist.init_process_group("nccl")
  mlflow.set_experiment(f"/Users/yang.yang@databricks.com/ConvaTec Workshop Computer Vision/Experiments_YOLO_CoCo")

  with mlflow.start_run() as run:  

    model = YOLO(f"{volume_project_location}/yolov8n.pt")
    model.train(
        batch=8,
        device=[0, 1],
        data="./coco8.yaml", # ref: https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco8.yaml
        epochs=50,
        project=f'{tmp_project_location}',
        exist_ok=True,
        fliplr=1,
        flipud=1,
        perspective=0.001,
        degrees=.45
    )
    # mlflow.log_params(vars(model.trainer.model.args))
    yolo_wrapper = YOLOC(model.trainer.best)
    mlflow.pyfunc.log_model(
        artifact_path="model",
        artifacts={'model_path': str(model.trainer.save_dir), "best_point": str(model.trainer.best)},
        python_model=yolo_wrapper,
        signature=signature
    )


distributor = TorchDistributor(num_processes=2, local_mode=True, use_gpu=True )
distributor.run(train_fn)

# pass. but saved to 3 different runs and under 2 experiments, 1 for 2, 1 for 1.

# COMMAND ----------

# DBTITLE 1,version K, still 3 different runs
from pyspark.ml.torch.distributor import TorchDistributor
mlflow.end_run()
# mlflow.set_experiment(f"/Users/yang.yang@databricks.com/ConvaTec Workshop Computer Vision/Experiments_YOLO_CoCo")

def train_fn():

  from ultralytics import YOLO
  import torch
  import mlflow
  import torch.distributed as dist
  from ultralytics import settings
  from mlflow.types.schema import Schema, ColSpec
  from mlflow.models.signature import ModelSignature

  # if not dist.is_initialized():
  #   # import torch.distributed as dist
  #   dist.init_process_group("nccl")

  model = YOLO(f"{volume_project_location}/yolov8n.pt")

  mlflow.set_experiment(f"/Users/yang.yang@databricks.com/ConvaTec Workshop Computer Vision/Experiments_YOLO_CoCo")

  with mlflow.start_run() as run:  

    model.train(
        batch=8,
        device=[0, 1],
        data="./coco8.yaml", # ref: https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco8.yaml
        epochs=50,
        project=f'{tmp_project_location}',
        exist_ok=True,
        fliplr=1,
        flipud=1,
        perspective=0.001,
        degrees=.45
    )
    # mlflow.log_params(vars(model.trainer.model.args))
    yolo_wrapper = YOLOC(model.trainer.best)
    mlflow.pyfunc.log_model(
        artifact_path="model",
        artifacts={'model_path': str(model.trainer.save_dir), "best_point": str(model.trainer.best)},
        python_model=yolo_wrapper,
        signature=signature
    )


distributor = TorchDistributor(num_processes=2, local_mode=True, use_gpu=True )
distributor.run(train_fn)

# pass. but still saved to 3 different runs and under 2 experiments, 1 for 2, 1 for 1.

# COMMAND ----------

# DBTITLE 1,version L, fail, mlflow naming issue due to volume path
from pyspark.ml.torch.distributor import TorchDistributor
mlflow.end_run()
# mlflow.set_experiment(f"/Users/yang.yang@databricks.com/ConvaTec Workshop Computer Vision/Experiments_YOLO_CoCo")

def train_fn():

  from ultralytics import YOLO
  import torch
  import mlflow
  import torch.distributed as dist
  from ultralytics import settings
  from mlflow.types.schema import Schema, ColSpec
  from mlflow.models.signature import ModelSignature

  # if not dist.is_initialized():
  #   # import torch.distributed as dist
  #   dist.init_process_group("nccl")

  model = YOLO(f"{volume_project_location}/yolov8n.pt")

  mlflow.set_experiment(f"/Users/yang.yang@databricks.com/ConvaTec Workshop Computer Vision/Experiments_YOLO_CoCo")

  with mlflow.start_run() as run:  

    model.train(
        batch=8,
        device=[0, 1],
        data="./coco8.yaml", # ref: https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco8.yaml
        epochs=5,
        project=f'{volume_project_location}',
        exist_ok=True,
        fliplr=1,
        flipud=1,
        perspective=0.001,
        degrees=.45
    )
    # mlflow.log_params(vars(model.trainer.model.args))
    yolo_wrapper = YOLOC(model.trainer.best)
    mlflow.pyfunc.log_model(
        artifact_path="model",
        artifacts={'model_path': str(model.trainer.save_dir), "best_point": str(model.trainer.best)},
        python_model=yolo_wrapper,
        signature=signature
    )


distributor = TorchDistributor(num_processes=2, local_mode=True, use_gpu=True )
distributor.run(train_fn)

#  mlflow.exceptions.RestException: RESOURCE_DOES_NOT_EXIST: Parent directory /Volumes/yyang/computer_vision/yolo does not exist. name issue

# COMMAND ----------

# DBTITLE 1,version M, fail, mlflow naming issue
from pyspark.ml.torch.distributor import TorchDistributor
mlflow.end_run()
mlflow.set_experiment(f"/Users/yang.yang@databricks.com/ConvaTec Workshop Computer Vision/Experiments_YOLO_CoCo")

def train_fn():

    from ultralytics import YOLO
    import torch
    import mlflow
    import torch.distributed as dist
    from ultralytics import settings
    from mlflow.types.schema import Schema, ColSpec
    from mlflow.models.signature import ModelSignature

    # if not dist.is_initialized():
    #   # import torch.distributed as dist
    #   dist.init_process_group("nccl")

    model = YOLO(f"{volume_project_location}/yolov8n.pt")


    model.train(
        batch=8,
        device=[0, 1],
        data="./coco8.yaml", # ref: https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco8.yaml
        epochs=5,
        project=f'{volume_project_location}',
        exist_ok=True,
        fliplr=1,
        flipud=1,
        perspective=0.001,
        degrees=.45
    )
    # mlflow.log_params(vars(model.trainer.model.args))
    yolo_wrapper = YOLOC(model.trainer.best)
    mlflow.pyfunc.log_model(
        artifact_path="model",
        artifacts={'model_path': str(model.trainer.save_dir), "best_point": str(model.trainer.best)},
        python_model=yolo_wrapper,
        signature=signature
    )


distributor = TorchDistributor(num_processes=2, local_mode=True, use_gpu=True )

mlflow.set_experiment(f"/Users/yang.yang@databricks.com/ConvaTec Workshop Computer Vision/Experiments_YOLO_CoCo")
with mlflow.start_run() as run: 
    distributor.run(train_fn)

# same, mlflow.exceptions.RestException: RESOURCE_DOES_NOT_EXIST: Parent directory /Volumes/yyang/computer_vision/yolo does not exist. name issue

# COMMAND ----------

# DBTITLE 1,version N, for rank 1, no experiment name set
from pyspark.ml.torch.distributor import TorchDistributor
mlflow.end_run()
mlflow.set_experiment(f"/Users/yang.yang@databricks.com/ConvaTec Workshop Computer Vision/Experiments_YOLO_CoCo")

settings.update({"mlflow":False}) # if you do want to autolog.
mlflow.autolog(disable = False)

def train_fn():

    from ultralytics import YOLO
    import torch
    import mlflow
    import torch.distributed as dist
    from ultralytics import settings
    from mlflow.types.schema import Schema, ColSpec
    from mlflow.models.signature import ModelSignature

    # if not dist.is_initialized():
    #   # import torch.distributed as dist
    #   dist.init_process_group("nccl")

    model = YOLO(f"{volume_project_location}/yolov8n.pt")


    model.train(
        batch=8,
        device=[0, 1],
        data="./coco8.yaml", # ref: https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco8.yaml
        epochs=5,
        project=f'{tmp_project_location}',
        exist_ok=True,
        fliplr=1,
        flipud=1,
        perspective=0.001,
        degrees=.45
    )
    # mlflow.log_params(vars(model.trainer.model.args))
    yolo_wrapper = YOLOC(model.trainer.best)
    mlflow.pyfunc.log_model(
        artifact_path="model",
        artifacts={'model_path': str(model.trainer.save_dir), "best_point": str(model.trainer.best)},
        python_model=yolo_wrapper,
        signature=signature
    )


distributor = TorchDistributor(num_processes=2, local_mode=True, use_gpu=True )

mlflow.set_experiment(f"/Users/yang.yang@databricks.com/ConvaTec Workshop Computer Vision/Experiments_YOLO_CoCo")
with mlflow.start_run() as run: 
    distributor.run(train_fn)

# [rank1]: mlflow.exceptions.RestException: RESOURCE_DOES_NOT_EXIST: No experiment was found. If using the Python fluent API, you can set an active experiment under which to create runs by calling mlflow.set_experiment("experiment_name") at the start of your program.

# COMMAND ----------

print(settings)

# COMMAND ----------

# DBTITLE 1,version O, passed, 1 exp 4 runs
from pyspark.ml.torch.distributor import TorchDistributor

settings.update({"mlflow":True}) # if you do want to autolog.
mlflow.autolog(disable = False)

def train_fn():

    from ultralytics import YOLO
    import torch
    import mlflow
    import torch.distributed as dist
    from ultralytics import settings
    from mlflow.types.schema import Schema, ColSpec
    from mlflow.models.signature import ModelSignature

    # if not dist.is_initialized():
    #   # import torch.distributed as dist
    #   dist.init_process_group("nccl")

    model = YOLO(f"{volume_project_location}/yolov8n.pt")


    model.train(
        batch=8,
        device=[0, 1],
        data="./coco8.yaml", # ref: https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco8.yaml
        epochs=5,
        project=f'{tmp_project_location}',
        exist_ok=True,
        fliplr=1,
        flipud=1,
        perspective=0.001,
        degrees=.45
    )
    # mlflow.log_params(vars(model.trainer.model.args))
    yolo_wrapper = YOLOC(model.trainer.best)
    mlflow.pyfunc.log_model(
        artifact_path="model",
        artifacts={'model_path': str(model.trainer.save_dir), "best_point": str(model.trainer.best)},
        python_model=yolo_wrapper,
        signature=signature
    )



mlflow.end_run()
experiment_name = "/Users/yang.yang@databricks.com/ConvaTec Workshop Computer Vision/Experiments_YOLO_CoCo"
mlflow.set_experiment(experiment_name)
experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id

os.environ['MLFLOW_EXPERIMENT_NAME'] = experiment_name
os.environ['MLFLOW_EXPERIMENT_ID'] = experiment_id
os.environ['MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING'] = "true"


with mlflow.start_run(experiment_id = experiment_id) as run: 
    distributor = TorchDistributor(num_processes=2, local_mode=True, use_gpu=True )
    distributor.run(train_fn)


# all works under 1 same experiment, but have 4 runs!

# COMMAND ----------

# DBTITLE 1,version P, almost there
from pyspark.ml.torch.distributor import TorchDistributor

settings.update({"mlflow":True}) # if you do want to autolog.
mlflow.autolog(disable = False)

def train_fn():

    from ultralytics import YOLO
    import torch
    import mlflow
    import torch.distributed as dist
    from ultralytics import settings
    from mlflow.types.schema import Schema, ColSpec
    from mlflow.models.signature import ModelSignature

    # if not dist.is_initialized():
    #   # import torch.distributed as dist
    #   dist.init_process_group("nccl")

    model = YOLO(f"{volume_project_location}/yolov8n.pt")


    model.train(
        batch=8,
        device=[0, 1],
        data="./coco8.yaml", # ref: https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco8.yaml
        epochs=5,
        project=f'{tmp_project_location}',
        exist_ok=True,
        fliplr=1,
        flipud=1,
        perspective=0.001,
        degrees=.45
    )
    # mlflow.log_params(vars(model.trainer.model.args))
    yolo_wrapper = YOLOC(model.trainer.best)
    mlflow.pyfunc.log_model(
        artifact_path="model",
        artifacts={'model_path': str(model.trainer.save_dir), "best_point": str(model.trainer.best)},
        python_model=yolo_wrapper,
        signature=signature
    )



mlflow.end_run()
experiment_name = "/Users/yang.yang@databricks.com/ConvaTec Workshop Computer Vision/Experiments_YOLO_CoCo"
mlflow.set_experiment(experiment_name)
experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id

# os.environ['MLFLOW_EXPERIMENT_NAME'] = experiment_name
os.environ['MLFLOW_EXPERIMENT_ID'] = experiment_id
os.environ['MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING'] = "true"


with mlflow.start_run(experiment_id = experiment_id) as run: 

    print("Active run_id: {}".format(run.info.run_id))
    os.environ['MLFLOW_RUN_ID'] = run.info.run_id # try to register under the same run

    distributor = TorchDistributor(num_processes=2, local_mode=True, use_gpu=True )
    distributor.run(train_fn)


# all works under 1 same experiment, but have 2 runs! much better!!

# COMMAND ----------

# DBTITLE 1,version Q, still 2 runs
from pyspark.ml.torch.distributor import TorchDistributor

settings.update({"mlflow":True}) # if you do want to autolog.
mlflow.autolog(disable = False)

def train_fn():

    from ultralytics import YOLO
    import torch
    import mlflow
    import torch.distributed as dist
    from ultralytics import settings
    from mlflow.types.schema import Schema, ColSpec
    from mlflow.models.signature import ModelSignature

    # if not dist.is_initialized():
    #   # import torch.distributed as dist
    #   dist.init_process_group("nccl")

    model = YOLO(f"{volume_project_location}/yolov8n.pt")


    model.train(
        batch=8,
        device=[0, 1],
        data="./coco8.yaml", # ref: https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco8.yaml
        epochs=5,
        project=f'{tmp_project_location}',
        exist_ok=True,
        fliplr=1,
        flipud=1,
        perspective=0.001,
        degrees=.45
    )
    # mlflow.log_params(vars(model.trainer.model.args))
    yolo_wrapper = YOLOC(model.trainer.best)
    mlflow.pyfunc.log_model(
        artifact_path="model",
        artifacts={'model_path': str(model.trainer.save_dir), "best_point": str(model.trainer.best)},
        python_model=yolo_wrapper,
        signature=signature
    )



mlflow.end_run()
experiment_name = "/Users/yang.yang@databricks.com/ConvaTec Workshop Computer Vision/Experiments_YOLO_CoCo"
mlflow.set_experiment(experiment_name)
experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id

# os.environ['MLFLOW_EXPERIMENT_NAME'] = experiment_name
os.environ['MLFLOW_EXPERIMENT_ID'] = experiment_id
os.environ['MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING'] = "true"

run = mlflow.start_run(experiment_id = experiment_id)
print("Active run_id: {}".format(run.info.run_id))
os.environ['MLFLOW_RUN_ID'] = run.info.run_id

with mlflow.start_run(run_id = run.info.run_id, nested=True) as run:     
    distributor = TorchDistributor(num_processes=2, local_mode=True, use_gpu=True )
    distributor.run(train_fn)


# still 2 runs.

# COMMAND ----------

# DBTITLE 1,version R, 4 runs, bad
from pyspark.ml.torch.distributor import TorchDistributor

settings.update({"mlflow":True}) # if you do want to autolog.
mlflow.autolog(disable = False)

def train_fn():

    from ultralytics import YOLO
    import torch
    import mlflow
    import torch.distributed as dist
    from ultralytics import settings
    from mlflow.types.schema import Schema, ColSpec
    from mlflow.models.signature import ModelSignature

    # if not dist.is_initialized():
    #   # import torch.distributed as dist
    #   dist.init_process_group("nccl")

    model = YOLO(f"{volume_project_location}/yolov8n.pt")


    model.train(
        batch=8,
        device=[0, 1],
        data="./coco8.yaml", # ref: https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco8.yaml
        epochs=5,
        project=f'{tmp_project_location}',
        exist_ok=True,
        fliplr=1,
        flipud=1,
        perspective=0.001,
        degrees=.45
    )
    # mlflow.log_params(vars(model.trainer.model.args))
    yolo_wrapper = YOLOC(model.trainer.best)
    mlflow.pyfunc.log_model(
        artifact_path="model",
        artifacts={'model_path': str(model.trainer.save_dir), "best_point": str(model.trainer.best)},
        python_model=yolo_wrapper,
        signature=signature
    )



mlflow.end_run()
experiment_name = "/Users/yang.yang@databricks.com/ConvaTec Workshop Computer Vision/Experiments_YOLO_CoCo"
mlflow.set_experiment(experiment_name)
experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id

# os.environ['MLFLOW_EXPERIMENT_NAME'] = experiment_name
os.environ['MLFLOW_EXPERIMENT_ID'] = experiment_id
os.environ['MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING'] = "true"

run = mlflow.start_run(experiment_id = experiment_id)
print("Active run_id: {}".format(run.info.run_id))
os.environ['MLFLOW_RUN_ID'] = run.info.run_id
mlflow.end_run()

with mlflow.start_run(run_id = run.info.run_id) as run:     
    distributor = TorchDistributor(num_processes=2, local_mode=True, use_gpu=True )
    distributor.run(train_fn)


# become 4 runs, bad!

# COMMAND ----------

# DBTITLE 1,version S, Almost Best Practice
from pyspark.ml.torch.distributor import TorchDistributor

settings.update({"mlflow":True}) # if you do want to autolog.
mlflow.autolog(disable = True)

def train_fn():

    from ultralytics import YOLO
    import torch
    import mlflow
    import torch.distributed as dist
    from ultralytics import settings
    from mlflow.types.schema import Schema, ColSpec
    from mlflow.models.signature import ModelSignature

    # if not dist.is_initialized():
    #   # import torch.distributed as dist
    #   dist.init_process_group("nccl")

    model = YOLO(f"{project_location}/raw_model/yolov8n.pt")


    model.train(
        batch=8,
        device=[0, 1],
        data="./coco8.yaml", # ref: https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco8.yaml
        epochs=5,
        project=f'{tmp_project_location}',
        exist_ok=True,
        fliplr=1,
        flipud=1,
        perspective=0.001,
        degrees=.45
    )
    # mlflow.log_params(vars(model.trainer.model.args))
    yolo_wrapper = YOLOC(model.trainer.best)
    mlflow.pyfunc.log_model(
        artifact_path="model",
        artifacts={'model_path': str(model.trainer.save_dir), "best_point": str(model.trainer.best)},
        python_model=yolo_wrapper,
        signature=signature
    )


mlflow.end_run()
experiment_name = "/Users/yang.yang@databricks.com/ConvaTec Workshop Computer Vision/Experiments_YOLO_CoCo"
mlflow.set_experiment(experiment_name)
experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id

# os.environ['MLFLOW_EXPERIMENT_NAME'] = experiment_name
os.environ['MLFLOW_EXPERIMENT_ID'] = experiment_id
os.environ['MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING'] = "true"


with mlflow.start_run(experiment_id = experiment_id) as run: 

    print("Active run_id: {}".format(run.info.run_id))
    os.environ['MLFLOW_RUN_ID'] = run.info.run_id # try to register under the same run

    distributor = TorchDistributor(num_processes=2, local_mode=True, use_gpu=True )
    distributor.run(train_fn)


# 2 runs, but it should be good enough to be considered "Almost - Best Practice".

# COMMAND ----------

# DBTITLE 1,version T, 1 run, no model metrics & paramters logged
from pyspark.ml.torch.distributor import TorchDistributor

settings.update({"mlflow":False}) # dont autolog
mlflow.autolog(disable = False)

def train_fn():

    from ultralytics import YOLO
    import torch
    import mlflow
    import torch.distributed as dist
    from ultralytics import settings
    from mlflow.types.schema import Schema, ColSpec
    from mlflow.models.signature import ModelSignature

    # if not dist.is_initialized():
    #   # import torch.distributed as dist
    #   dist.init_process_group("nccl")

    model = YOLO(f"{project_location}/raw_model/yolov8n.pt")


    model.train(
        batch=8,
        device=[0, 1],
        data="./coco8.yaml", # ref: https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco8.yaml
        epochs=5,
        project=f'{tmp_project_location}',
        exist_ok=True,
        fliplr=1,
        flipud=1,
        perspective=0.001,
        degrees=.45
    )
    # mlflow.log_params(vars(model.trainer.model.args))
    yolo_wrapper = YOLOC(model.trainer.best)
    mlflow.pyfunc.log_model(
        artifact_path="model",
        artifacts={'model_path': str(model.trainer.save_dir), "best_point": str(model.trainer.best)},
        python_model=yolo_wrapper,
        signature=signature
    )


mlflow.end_run()
experiment_name = "/Users/yang.yang@databricks.com/ConvaTec Workshop Computer Vision/Experiments_YOLO_CoCo"
mlflow.set_experiment(experiment_name)
experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id

# os.environ['MLFLOW_EXPERIMENT_NAME'] = experiment_name
os.environ['MLFLOW_EXPERIMENT_ID'] = experiment_id
os.environ['MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING'] = "true"


with mlflow.start_run(experiment_id = experiment_id) as run: 

    print("Active run_id: {}".format(run.info.run_id))
    os.environ['MLFLOW_RUN_ID'] = run.info.run_id # try to register under the same run

    distributor = TorchDistributor(num_processes=2, local_mode=True, use_gpu=True )
    distributor.run(train_fn)


# disable Ultra side autolog + enable MLflow side autolog + log Model.
# results: 1 run! However, no model metrics, no model paramters, as we didnt' specify manually above.

# COMMAND ----------

# DBTITLE 1,version U, 2 runs, same as enable both logging
from pyspark.ml.torch.distributor import TorchDistributor

settings.update({"mlflow":True})
mlflow.autolog(disable = True)

def train_fn():

    from ultralytics import YOLO
    import torch
    import mlflow
    import torch.distributed as dist
    from ultralytics import settings
    from mlflow.types.schema import Schema, ColSpec
    from mlflow.models.signature import ModelSignature

    # if not dist.is_initialized():
    #   # import torch.distributed as dist
    #   dist.init_process_group("nccl")

    model = YOLO(f"{project_location}/raw_model/yolov8n.pt")


    model.train(
        batch=8,
        device=[0, 1],
        data="./coco8.yaml", # ref: https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco8.yaml
        epochs=5,
        project=f'{tmp_project_location}',
        exist_ok=True,
        fliplr=1,
        flipud=1,
        perspective=0.001,
        degrees=.45
    )
    # mlflow.log_params(vars(model.trainer.model.args))
    yolo_wrapper = YOLOC(model.trainer.best)
    mlflow.pyfunc.log_model(
        artifact_path="model",
        artifacts={'model_path': str(model.trainer.save_dir), "best_point": str(model.trainer.best)},
        python_model=yolo_wrapper,
        signature=signature
    )


mlflow.end_run()
experiment_name = "/Users/yang.yang@databricks.com/ConvaTec Workshop Computer Vision/Experiments_YOLO_CoCo"
mlflow.set_experiment(experiment_name)
experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id

# os.environ['MLFLOW_EXPERIMENT_NAME'] = experiment_name
os.environ['MLFLOW_EXPERIMENT_ID'] = experiment_id
os.environ['MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING'] = "true"


with mlflow.start_run(experiment_id = experiment_id) as run: 

    print("Active run_id: {}".format(run.info.run_id))
    os.environ['MLFLOW_RUN_ID'] = run.info.run_id # try to register under the same run

    distributor = TorchDistributor(num_processes=2, local_mode=True, use_gpu=True )
    distributor.run(train_fn)


# Enable Ultra side autolog + disable MLflow side autolog + log Model.
# results: 2 run, same as enable both logging as MLflow side autolog doesn't do anything really.

# COMMAND ----------

# DBTITLE 1,version V, 1 run, no model logged
from pyspark.ml.torch.distributor import TorchDistributor

settings.update({"mlflow":True})
mlflow.autolog(disable = False, log_models = True)

def train_fn():

    from ultralytics import YOLO
    import torch
    import mlflow
    import torch.distributed as dist
    from ultralytics import settings
    from mlflow.types.schema import Schema, ColSpec
    from mlflow.models.signature import ModelSignature

    # if not dist.is_initialized():
    #   # import torch.distributed as dist
    #   dist.init_process_group("nccl")

    model = YOLO(f"{project_location}/raw_model/yolov8n.pt")


    model.train(
        batch=8,
        device=[0, 1],
        data="./coco8.yaml", # ref: https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco8.yaml
        epochs=5,
        project=f'{tmp_project_location}',
        exist_ok=True,
        fliplr=1,
        flipud=1,
        perspective=0.001,
        degrees=.45
    )


mlflow.end_run()
experiment_name = "/Users/yang.yang@databricks.com/ConvaTec Workshop Computer Vision/Experiments_YOLO_CoCo"
mlflow.set_experiment(experiment_name)
experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id

# os.environ['MLFLOW_EXPERIMENT_NAME'] = experiment_name
os.environ['MLFLOW_EXPERIMENT_ID'] = experiment_id
os.environ['MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING'] = "true"


with mlflow.start_run(experiment_id = experiment_id) as run: 

    print("Active run_id: {}".format(run.info.run_id))
    os.environ['MLFLOW_RUN_ID'] = run.info.run_id # try to register under the same run

    distributor = TorchDistributor(num_processes=2, local_mode=True, use_gpu=True )
    distributor.run(train_fn)


# Enable Ultra side autolog + Enable MLflow side autolog + no manually log Model.
# results: 1 run, model didn't logged even with both autolog enabled, since it really needs PyFuncModel to be able to log.

# COMMAND ----------

# DBTITLE 1,version W, 1 run, need to drilldow model attributes to find out hyperparameters and metrics
from pyspark.ml.torch.distributor import TorchDistributor

settings.update({"mlflow":False}) # dont autolog
mlflow.autolog(disable = False)

def train_fn():

    from ultralytics import YOLO
    import torch
    import mlflow
    import torch.distributed as dist
    from ultralytics import settings
    from mlflow.types.schema import Schema, ColSpec
    from mlflow.models.signature import ModelSignature

    # if not dist.is_initialized():
    #   # import torch.distributed as dist
    #   dist.init_process_group("nccl")

    model = YOLO(f"{project_location}/raw_model/yolov8n.pt")


    model.train(
        batch=8,
        device=[0, 1],
        data="./coco8.yaml", # ref: https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco8.yaml
        epochs=5,
        project=f'{tmp_project_location}',
        exist_ok=True,
        fliplr=1,
        flipud=1,
        perspective=0.001,
        degrees=.45
    )
    # mlflow.log_params(vars(model.trainer.model.args))
    mlflow.log_params(vars(model.args))
    for metric_name, metric_value in model.trainer.metrics.items():
        mlflow.log_metric(metric_name, metric_value)
    yolo_wrapper = YOLOC(model.trainer.best)
    mlflow.pyfunc.log_model(
        artifact_path="model",
        artifacts={'model_path': str(model.trainer.save_dir), "best_point": str(model.trainer.best)},
        python_model=yolo_wrapper,
        signature=signature
    )


mlflow.end_run()
experiment_name = "/Users/yang.yang@databricks.com/ConvaTec Workshop Computer Vision/Experiments_YOLO_CoCo"
mlflow.set_experiment(experiment_name)
experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id

# os.environ['MLFLOW_EXPERIMENT_NAME'] = experiment_name
os.environ['MLFLOW_EXPERIMENT_ID'] = experiment_id
os.environ['MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING'] = "true"


with mlflow.start_run(experiment_id = experiment_id) as run: 

    print("Active run_id: {}".format(run.info.run_id))
    os.environ['MLFLOW_RUN_ID'] = run.info.run_id # try to register under the same run

    distributor = TorchDistributor(num_processes=2, local_mode=True, use_gpu=True )
    distributor.run(train_fn)


# syntax wrong for manually logging model hyperparams and metrics, need more drilldown give more time
# [rank1]: AttributeError: 'YOLO' object has no attribute 'args'

# COMMAND ----------

# DBTITLE 1,Version X, distinguished rank logging mechanism
from pyspark.ml.torch.distributor import TorchDistributor

settings.update({"mlflow":True}) # if you do want to autolog.
mlflow.autolog(disable = False)

def train_fn():

    from ultralytics import YOLO
    import torch
    import mlflow
    import torch.distributed as dist
    from ultralytics import settings
    from mlflow.types.schema import Schema, ColSpec
    from mlflow.models.signature import ModelSignature

    model = YOLO(f"{project_location}/raw_model/yolov8n.pt")

    model.train(
        batch=8,
        device=[0, 1],
        data="./coco8.yaml", # ref: https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco8.yaml
        epochs=5,
        project=f'{tmp_project_location}',
        exist_ok=True,
        fliplr=1,
        flipud=1,
        perspective=0.001,
        degrees=.45
    )

    # after training is done.
    if not dist.is_initialized():
      # import torch.distributed as dist
      dist.init_process_group("nccl")
    rank = dist.get_rank()

    mlflow.log_params({"rank":rank})


    if rank == 1:
      yolo_wrapper = YOLOC(model.trainer.best)
      mlflow.pyfunc.log_model(
          artifact_path="model",
          artifacts={'model_path': str(model.trainer.save_dir), "best_point": str(model.trainer.best)},
          python_model=yolo_wrapper,
          signature=signature
      )


mlflow.end_run()
experiment_name = "/Users/yang.yang@databricks.com/ConvaTec Workshop Computer Vision/Experiments_YOLO_CoCo"
mlflow.set_experiment(experiment_name)
experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id

# os.environ['MLFLOW_EXPERIMENT_NAME'] = experiment_name
os.environ['MLFLOW_EXPERIMENT_ID'] = experiment_id
os.environ['MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING'] = "true"


with mlflow.start_run(experiment_id = experiment_id) as run: 

    print("Active run_id: {}".format(run.info.run_id))
    os.environ['MLFLOW_RUN_ID'] = run.info.run_id # try to register under the same run

    distributor = TorchDistributor(num_processes=2, local_mode=True, use_gpu=True )
    distributor.run(train_fn)


# 2 runs, now we can distinguish one run logged by rank 0 and one run logged by rank 1 GPU. If given only 2 GPUs, ultralytics will use rank 1 (max I assume)'s run for autologging. Therefore, we just need to merge every other logging artifacts from rank 0 to rank 1, so we evetually will just have 1 single run by rank 1. This will be tested in next version Y.

# COMMAND ----------

num_gpus = torch.cuda.device_count()
list(range(num_gpus))

# COMMAND ----------

# Set the tracking URI to Databricks
mlflow.set_tracking_uri("databricks")

# Set the workspace ID and personal access token
workspace_id = "984752964297111"

# COMMAND ----------

# DBTITLE 1,Version Y, figuring out more 2 GPUs vs 4 GPUs!
from pyspark.ml.torch.distributor import TorchDistributor

settings.update({"mlflow":True}) # if you do want to autolog.
mlflow.autolog(disable = False)


def train_fn(device_list = [0,1,2,3]):

    from ultralytics import YOLO
    import torch
    import mlflow
    import torch.distributed as dist
    from ultralytics import settings
    from mlflow.types.schema import Schema, ColSpec
    from mlflow.models.signature import ModelSignature

    model = YOLO(f"{project_location}/raw_model/yolov8n.pt")

    model.train(
        batch=8,
        device=device_list,
        data="./coco8.yaml", # ref: https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco8.yaml
        epochs=5,
        project=f'{tmp_project_location}',
        exist_ok=True,
        fliplr=1,
        flipud=1,
        perspective=0.001,
        degrees=.45
    )

    # after training is done.
    if not dist.is_initialized():
      # import torch.distributed as dist
      dist.init_process_group("nccl")
    rank = dist.get_rank()

    # we divert all other manual logging to the last GPU with max rank.
    # if rank == torch.cuda.device_count() - 1:
    # if rank == 0:

    mlflow.log_params({"rank":rank})
    yolo_wrapper = YOLOC(model.trainer.best)
    mlflow.pyfunc.log_model(
        artifact_path="model",
        artifacts={'model_path': str(model.trainer.save_dir), "best_point": str(model.trainer.best)},
        python_model=yolo_wrapper,
        signature=signature
      )


mlflow.end_run()
experiment_name = "/Users/yang.yang@databricks.com/ConvaTec Workshop Computer Vision/Experiments_YOLO_CoCo"
mlflow.set_experiment(experiment_name)
experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id

# os.environ['MLFLOW_EXPERIMENT_NAME'] = experiment_name
os.environ['MLFLOW_EXPERIMENT_ID'] = experiment_id
os.environ['MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING'] = "true"


# Reset MLFLOW_RUN_ID
if 'MLFLOW_RUN_ID' in os.environ:
    del os.environ['MLFLOW_RUN_ID']
with mlflow.start_run(experiment_id = experiment_id) as run: 

    # print("Active run_id: {}".format(run.info.run_id))
    # os.environ['MLFLOW_RUN_ID'] = run.info.run_id # try to register under the same run

    num_gpus = torch.cuda.device_count()
    device_list = list(range(num_gpus))
    print("num_gpus:", num_gpus)
    print("device_list:", device_list)


    distributor = TorchDistributor(num_processes=num_gpus, local_mode=True, use_gpu=True)
    distributor.run(train_fn, device_list = device_list)


# 2 GPU works, but not for 4, 4 GPUs confused run id again.

# COMMAND ----------



# COMMAND ----------

# DBTITLE 1,Version AA, Very Bad, have 5 Runs produced
from pyspark.ml.torch.distributor import TorchDistributor

settings.update({"mlflow":True}) # if you do want to autolog.
mlflow.autolog(disable = False)


def train_fn(device_list = [0,1,2,3]):

    from ultralytics import YOLO
    import torch
    import mlflow
    import torch.distributed as dist
    from ultralytics import settings
    from mlflow.types.schema import Schema, ColSpec
    from mlflow.models.signature import ModelSignature

    ############################

    ##### Setting up MLflow ####
    # We need to do this so that different processes that will be able to find mlflow
    os.environ['DATABRICKS_HOST'] = db_host
    os.environ['DATABRICKS_TOKEN'] = db_token
    os.environ['MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING'] = "true"
    os.environ['MLFLOW_EXPERIMENT_NAME'] = experiment_name
    # We set the experiment details here
    experiment = mlflow.set_experiment(experiment_name)
    
    #
    model = YOLO(f"{project_location}/raw_model/yolov8n.pt")

    model.train(
        batch=8,
        device=device_list,
        data="./coco8.yaml", # ref: https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco8.yaml
        epochs=5,
        project=f'{tmp_project_location}',
        exist_ok=True,
        fliplr=1,
        flipud=1,
        perspective=0.001,
        degrees=.45
    )

          
    yolo_wrapper = YOLOC(model.trainer.best)
    mlflow.pyfunc.log_model(
        artifact_path="model",
        artifacts={'model_path': str(model.trainer.save_dir), "best_point": str(model.trainer.best)},
        python_model=yolo_wrapper,
        signature=signature
    )

    if dist.is_initialized():
        dist.destroy_process_group()

    return "finished" # can return any picklable object


mlflow.end_run()
experiment_name = "/Users/yang.yang@databricks.com/ConvaTec Workshop Computer Vision/Experiments_YOLO_CoCo"
# mlflow.set_experiment(experiment_name)
# experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
# # os.environ['MLFLOW_EXPERIMENT_NAME'] = experiment_name
# os.environ['MLFLOW_EXPERIMENT_ID'] = experiment_id
# os.environ['MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING'] = "true"


# Reset MLFLOW_RUN_ID
if 'MLFLOW_RUN_ID' in os.environ:
    del os.environ['MLFLOW_RUN_ID']


num_gpus = torch.cuda.device_count()
device_list = list(range(num_gpus))
print("num_gpus:", num_gpus)
print("device_list:", device_list)


distributor = TorchDistributor(num_processes=num_gpus, local_mode=True, use_gpu=True)
distributor.run(train_fn, device_list = device_list)


# Very bad, 1 autorun + 4 gpus' manual log run id.

# COMMAND ----------

# DBTITLE 1,Version Z2, BP with nested worker runs
from pyspark.ml.torch.distributor import TorchDistributor

settings.update({"mlflow":True}) # if you do want to autolog.
mlflow.autolog(disable = False)
os.environ["CUDA_LAUNCH_BLOCKING"] = "0" # Reset to asynchronous execution for production in production

def train_fn(device_list = [0,1,2,3], parent_run_id = None):

    from ultralytics import YOLO
    import torch
    import mlflow
    import torch.distributed as dist
    from ultralytics import settings
    from mlflow.types.schema import Schema, ColSpec
    from mlflow.models.signature import ModelSignature

    ############################

    ##### Setting up MLflow ####
    # We need to do this so that different processes that will be able to find mlflow
    os.environ['DATABRICKS_HOST'] = db_host # pending replace with db vault secret
    os.environ['DATABRICKS_TOKEN'] = db_token # pending replace with db vault secret
    os.environ['MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING'] = "true"
    os.environ['MLFLOW_EXPERIMENT_NAME'] = experiment_name
    os.environ['DATABRICKS_WORKSPACE_ID'] = db_wksp_id  # Set the workspace ID
    # We set the experiment details here
    experiment = mlflow.set_experiment(experiment_name)
    
    #
    model = YOLO(f"{project_location}/raw_model/yolov8n.pt")
    model.train(
        batch=8,
        device=device_list,
        data="./coco8.yaml", # ref: https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco8.yaml
        epochs=5,
        project=f'{tmp_project_location}',
        exist_ok=True,
        fliplr=1,
        flipud=1,
        perspective=0.001,
        degrees=.45
    )

    # active_run_id = mlflow.last_active_run().info.run_id
    # print("For YOLO autologging, active_run_id is: ", active_run_id)

    # # after training is done.
    # if not dist.is_initialized():
    #   # import torch.distributed as dist
    #   dist.init_process_group("nccl")

    local_rank = int(os.environ["LOCAL_RANK"])
    global_rank = int(os.environ["RANK"])
    
    if global_rank == 0:

        with mlflow.start_run(run_id=parent_run_id) as run:
            mlflow.log_artifact("./coco8.yaml", "input_data_yaml")
            mlflow.log_dict(data, "data.yaml")
            mlflow.log_params({"rank":global_rank})
            yolo_wrapper = YOLOC(model.trainer.best)
            mlflow.pyfunc.log_model(
                artifact_path="model",
                artifacts={'model_path': str(model.trainer.save_dir), "best_point": str(model.trainer.best)},
                python_model=yolo_wrapper,
                signature=signature)

    # clean up
    if dist.is_initialized():
        dist.destroy_process_group()

    return "finished" # can return any picklable object


if __name__ == "__main__":
    mlflow.end_run()
    experiment_name = "/Users/yang.yang@databricks.com/ConvaTec Workshop Computer Vision/Experiments_YOLO_CoCo"
    os.environ['DATABRICKS_HOST'] = db_host # pending replace with db vault secret
    print(f"DATABRICKS_HOST set to {os.environ['DATABRICKS_HOST']}")
    os.environ['DATABRICKS_TOKEN'] = db_token # pending replace with db vault secret
    print(f"DATABRICKS_TOKEN set to {os.environ['DATABRICKS_TOKEN']}") # should be redacted
    os.environ['MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING'] = "true"
    print(f"MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING set to {os.environ['MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING']}")
    os.environ['MLFLOW_EXPERIMENT_NAME'] = experiment_name
    print(f"MLFLOW_EXPERIMENT_NAME set to {os.environ['MLFLOW_EXPERIMENT_NAME']}")
    os.environ['DATABRICKS_WORKSPACE_ID'] = db_wksp_id  # Set the workspace ID
    print(f"DATABRICKS_WORKSPACE_ID set to {os.environ['DATABRICKS_WORKSPACE_ID']}")

    mlflow.set_experiment(experiment_name)
    experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id

    # Reset MLFLOW_RUN_ID, so we dont bump into the wrong one.
    if 'MLFLOW_RUN_ID' in os.environ:
        del os.environ['MLFLOW_RUN_ID']

    num_gpus = torch.cuda.device_count()
    device_list = list(range(num_gpus))
    print("num_gpus:", num_gpus)
    print("device_list:", device_list)

    with mlflow.start_run(experiment_id=experiment_id) as ParentRun:
        active_run_id = mlflow.last_active_run().info.run_id
        active_run_name = mlflow.last_active_run().info.run_name

        print("For master triggering run, active_run_id is: ", active_run_id)
        print("For master triggering run, active_run_name is: ", active_run_name)
        print(f"For master triggering run, active_run_id is: '{active_run_id}' and active_run_name is: '{active_run_name}'.")
        print(f"All worker runs will be logged into the same run id '{active_run_id}' and name '{active_run_name}'.")

        with mlflow.start_run(run_name='ChildRun', nested=True) as ChildRun:
            distributor = TorchDistributor(num_processes=num_gpus, local_mode=True, use_gpu=True)      
            distributor.run(train_fn, device_list = device_list, parent_run_id = active_run_id)


# 
# 1. outter run id for recording system metrics only, more granular
# 2. Inner run id for auto-logging and manually logging other artifacts.

# Conclusion, nested runs are messy and ugly formated. not like this.

# COMMAND ----------

# DBTITLE 1,Version Z, this is the Best Practice found so far!!!
from pyspark.ml.torch.distributor import TorchDistributor

settings.update({"mlflow":True}) # if you do want to autolog.
mlflow.autolog(disable = False)


def train_fn(device_list = [0,1,2,3], active_run_id = None):

    from ultralytics import YOLO
    import torch
    import mlflow
    import torch.distributed as dist
    from ultralytics import settings
    from mlflow.types.schema import Schema, ColSpec
    from mlflow.models.signature import ModelSignature

    ############################

    ##### Setting up MLflow ####
    # We need to do this so that different processes that will be able to find mlflow
    os.environ['DATABRICKS_HOST'] = db_host # pending replace with db vault secret
    os.environ['DATABRICKS_TOKEN'] = db_token # pending replace with db vault secret 
    os.environ['MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING'] = "true"
    os.environ['MLFLOW_EXPERIMENT_NAME'] = experiment_name
    os.environ['DATABRICKS_WORKSPACE_ID'] = db_wksp_id  # Set the workspace ID
    # We set the experiment details here
    experiment = mlflow.set_experiment(experiment_name)
    
    #
    with mlflow.start_run(run_id=active_run_id) as run:
        model = YOLO(f"{project_location}/raw_model/yolov8n.pt")
        model.train(
            batch=8,
            device=device_list,
            data="./coco8.yaml", # ref: https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco8.yaml
            epochs=5,
            project=f'{tmp_project_location}',
            exist_ok=True,
            fliplr=1,
            flipud=1,
            perspective=0.001,
            degrees=.45
        )

    # active_run_id = mlflow.last_active_run().info.run_id
    # print("For YOLO autologging, active_run_id is: ", active_run_id)

    # # after training is done.
    # if not dist.is_initialized():
    #   # import torch.distributed as dist
    #   dist.init_process_group("nccl")

    local_rank = int(os.environ["LOCAL_RANK"])
    global_rank = int(os.environ["RANK"])
    
    if global_rank == 0:

        active_run_id = mlflow.last_active_run().info.run_id
        print("For YOLO autologging, active_run_id is: ", active_run_id)

        # Get the list of runs in the experiment
        runs = mlflow.search_runs(experiment_names=[experiment_name], order_by=["start_time DESC"], max_results=1)

        # Extract the latest run_id
        if not runs.empty:
            latest_run_id = runs.iloc[0].run_id
            print(f"Latest run_id: {latest_run_id}")
        else:
            print("No runs found in the experiment.")


        with mlflow.start_run(run_id=latest_run_id) as run:
            mlflow.log_artifact("./coco8.yaml", "input_data_yaml")
            mlflow.log_dict(data, "data.yaml")
            mlflow.log_params({"rank":global_rank})
            yolo_wrapper = YOLOC(model.trainer.best)
            mlflow.pyfunc.log_model(
                artifact_path="model",
                artifacts={'model_path': str(model.trainer.save_dir), "best_point": str(model.trainer.best)},
                python_model=yolo_wrapper,
                signature=signature)

    # clean up
    if dist.is_initialized():
        dist.destroy_process_group()

    return "finished" # can return any picklable object


if __name__ == "__main__":
    mlflow.end_run()
    experiment_name = "/Users/yang.yang@databricks.com/ConvaTec Workshop Computer Vision/Experiments_YOLO_CoCo"
    os.environ['DATABRICKS_HOST'] = db_host # pending replace with db vault secret
    print(f"DATABRICKS_HOST set to {os.environ['DATABRICKS_HOST']}")
    os.environ['DATABRICKS_TOKEN'] = db_token # pending replace with db vault secret
    print(f"DATABRICKS_TOKEN set to {os.environ['DATABRICKS_TOKEN']}") # should be redacted
    os.environ['MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING'] = "true"
    print(f"MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING set to {os.environ['MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING']}")
    os.environ['MLFLOW_EXPERIMENT_NAME'] = experiment_name
    print(f"MLFLOW_EXPERIMENT_NAME set to {os.environ['MLFLOW_EXPERIMENT_NAME']}")
    os.environ['DATABRICKS_WORKSPACE_ID'] = db_wksp_id  # Set the workspace ID
    print(f"DATABRICKS_WORKSPACE_ID set to {os.environ['DATABRICKS_WORKSPACE_ID']}")

    mlflow.set_experiment(experiment_name)
    experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id

    # Reset MLFLOW_RUN_ID, so we dont bump into the wrong one.
    if 'MLFLOW_RUN_ID' in os.environ:
        del os.environ['MLFLOW_RUN_ID']

    num_gpus = torch.cuda.device_count()
    device_list = list(range(num_gpus))
    print("num_gpus:", num_gpus)
    print("device_list:", device_list)

    with mlflow.start_run(experiment_id=experiment_id) as run:
        active_run_id = mlflow.last_active_run().info.run_id
        active_run_name = mlflow.last_active_run().info.run_name

        print("For master triggering run, active_run_id is: ", active_run_id)
        print("For master triggering run, active_run_name is: ", active_run_name)
        print(f"For master triggering run, active_run_id is: '{active_run_id}' and active_run_name is: '{active_run_name}'.")
        print(f"All worker runs will be logged into the same run id '{active_run_id}' and name '{active_run_name}'.")

        distributor = TorchDistributor(num_processes=num_gpus, local_mode=True, use_gpu=True)      
        distributor.run(train_fn, device_list = device_list, active_run_id = active_run_id)


# Best Practice You Are! Previously we have 2 runs, one for master and the other for all workers.
# 1. master run id for recording system metrics only
# 2. Inner run id for auto-logging and manually logging other artifacts.

# Now I have changed the code to merge outter run and inner run to be the same run id. less confusion.

# COMMAND ----------

# MAGIC %md
# MAGIC # Supplemental Testing

# COMMAND ----------

# DBTITLE 1,make sure always has brand new nccl connection
# #: NCCL connection is using server socket

# if dist.is_initialized():
#   dist.destroy_process_group() # make sure we dont double init_process_group, so destroy it first if any

# if not dist.is_initialized():
#   # import torch.distributed as dist
#   dist.init_process_group("nccl")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Testing a few storing options

# COMMAND ----------

# MAGIC %ls /Volumes/yyang/computer_vision/yolo/data/coco8/images/train/000000000009.jpg

# COMMAND ----------

dbfs_project_location

# COMMAND ----------

dbfs_project_location

# COMMAND ----------

volume_project_location

# COMMAND ----------

tmp_project_location

# COMMAND ----------

dbutils.fs.put(f"{volume_project_location}/test_file.txt", "This is a test file.", overwrite=True)

# COMMAND ----------

dbutils.fs.put(f"{dbfs_project_location}/test_file.txt", "This is a test file.", overwrite=True)

# COMMAND ----------

tmp_project_location

# COMMAND ----------

dbutils.fs.put(f"{tmp_project_location}/test_file.txt", "This is a test file.", overwrite=True)

# COMMAND ----------

# MAGIC %ls -ahlrt $dbfs_project_loction

# COMMAND ----------

dbutils.fs.ls(dbfs_project_location)

# COMMAND ----------

# MAGIC %ls -ahlrt $volume_project_location

# COMMAND ----------

# MAGIC %ls -ahlrt $volume_project_location/train

# COMMAND ----------

dbutils.fs.ls(volume_project_location)

# COMMAND ----------

# MAGIC %ls -ahlrt $tmp_project_location

# COMMAND ----------

dbutils.fs.ls(tmp_project_location)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Training starts

# COMMAND ----------

# model = YOLO("yolov8l-seg.pt")
model = YOLO("yolov8n.pt")

#: NCCL connection is using server socket
if dist.is_initialized():
  dist.destroy_process_group() # make sure we dont double init_process_group, so destroy it first if any

# if not dist.is_initialized():
#   # import torch.distributed as dist
#   dist.init_process_group("nccl")

model.train(
        batch=8,
        device=[0,1],
        # data=f"{project_location}/training/data.yaml",
        data=f"./coco8.yaml",
        epochs=50,
        # project='/tmp/solar_panel_damage/',
        # project=dbfs_project_location, # this will fail on the 2nd attempt to write into csv in appending mode
        # project=volume_project_location, # this will fail on the 2nd attempt to write into csv in appending mode
        project=tmp_project_location, # this will work as folder is ephemeral and without RBAC.
        exist_ok=True,
        fliplr=1,
        flipud=1,
        perspective=0.001,
        degrees=.45
    )

# COMMAND ----------

mlflow.end_run()
with mlflow.start_run():
  # model = YOLO("yolov8l-seg.pt")
  model = YOLO("yolov8n.pt")

  #: NCCL connection is using server socket
  if dist.is_initialized():
    dist.destroy_process_group() # make sure we dont double init_process_group, so destroy it first if any

  if not dist.is_initialized():
    # import torch.distributed as dist
    dist.init_process_group("nccl")

  model.train(
          batch=8,
          device=[0,1],
          # data=f"{project_location}/training/data.yaml",
          data=f"./coco8.yaml",
          epochs=50,
          # project='/tmp/solar_panel_damage/',
          # project=dbfs_project_location, # this will fail on the 2nd attempt to write into csv in appending mode
          # project=volume_project_location, # this will fail on the 2nd attempt to write into csv in appending mode
          project=tmp_project_location, # this will work as folder is ephemeral and without RBAC.
          exist_ok=True,
          fliplr=1,
          flipud=1,
          perspective=0.001,
          degrees=.45
      )

  mlflow.log_params(vars(model.trainer.model.args))
  yolo_wrapper = YOLOC(model.trainer.best) # creates an instance of YOLOC with the best model checkpoint.
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
      data="coco8.yaml", # ref: https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco8.yaml
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
