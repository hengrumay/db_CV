# Databricks notebook source
# MAGIC %md
# MAGIC ## MedCellTypes Instance Segmentation: 
# MAGIC
# MAGIC #### YOLO Training on SingleNode-MultiGPU Cluster on Databricks with MosaicAI Streaming Datasets 

# COMMAND ----------

# MAGIC %md
# MAGIC What is Instance Segmentation?
# MAGIC Instance segmentation goes beyond basic object detection, which draws bounding boxes around objects, and semantic segmentation, which labels each pixel in an image with a class but does not differentiate between individual objects of the same class. Instead, instance segmentation uniquely identifies each object instance, even when they overlap. For example, in an image with multiple cars, instance segmentation will not only recognize all of them as 'car' but will also create a separate, pixel-perfect mask for each individual car, distinguishing them from one another and the background. This capability is crucial in scenarios where counting individual objects or analyzing their specific shapes is important.
# MAGIC
# MAGIC Instance Segmentation vs. Related Tasks
# MAGIC While related, instance segmentation differs significantly from other computer vision tasks:
# MAGIC
# MAGIC - Object Detection: Object detection focuses on identifying and localizing objects within an image by drawing bounding boxes around them. It tells you what and where objects are, but not their exact shape or boundaries.
# MAGIC - Semantic Segmentation: Semantic segmentation classifies each pixel in an image into predefined classes, such as 'sky,' 'road,' or 'car.' It provides a pixel-level understanding of the scene but does not differentiate between separate instances of the same object class. For example, all cars are labeled as 'car' pixels, but are not distinguished as individual objects.
# MAGIC - Instance Segmentation: Instance segmentation combines the strengths of both. It performs pixel-level classification like semantic segmentation, but also differentiates and segments each object instance individually, like object detection, providing a comprehensive and detailed understanding of the objects in an image.

# COMMAND ----------

# MAGIC %md
# MAGIC Below image shows the example of **Instance Segmentation**.
# MAGIC
# MAGIC Specifically, this shows the Mosaiced Image: 
# MAGIC
# MAGIC This image demonstrates a training batch composed of mosaiced dataset images. Mosaicing is a technique used during training that combines multiple images into a single image to increase the variety of objects and scenes within each training batch. This aids the model's ability to generalize to different object sizes, aspect ratios, and contexts.
# MAGIC ![](https://github.com/ultralytics/docs/releases/download/0/mosaiced-training-batch-3.avif)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Training Framework Background README

# COMMAND ----------

# MAGIC %md
# MAGIC ### Distributed Pytorch
# MAGIC PyTorch distributed package supports Linux (stable), MacOS (stable), and Windows (prototype). By default for Linux, the Gloo and NCCL backends are built and included in PyTorch distributed (NCCL only when building with CUDA). MPI is an optional backend that can only be included if you build PyTorch from source. (e.g. building PyTorch on a host that has MPI installed.)
# MAGIC In short, 
# MAGIC - use NCCL for GPU,
# MAGIC - use Gloo for CPU,
# MAGIC - use MPI if Gloo wont work for CPU
# MAGIC
# MAGIC ref: https://pytorch.org/docs/stable/distributed.html

# COMMAND ----------

# MAGIC %md
# MAGIC ### Applying Parallelism To Scale Your Model
# MAGIC Data Parallelism is a widely adopted single-program multiple-data training paradigm where the model is replicated on every process, every model replica computes local gradients for a different set of input data samples, gradients are averaged within the data-parallel communicator group before each optimizer step.
# MAGIC
# MAGIC Model Parallelism techniques (or Sharded Data Parallelism) are required when a model doesn’t fit in GPU, and can be combined together to form multi-dimensional (N-D) parallelism techniques.
# MAGIC
# MAGIC When deciding what parallelism techniques to choose for your model, use these common guidelines:
# MAGIC
# MAGIC - Use DistributedDataParallel (DDP), if your model fits in a single GPU but you want to easily scale up training using multiple GPUs.
# MAGIC
# MAGIC   + Use torchrun, to launch multiple pytorch processes if you are using more than one node.
# MAGIC
# MAGIC   + See also: Getting Started with Distributed Data Parallel
# MAGIC
# MAGIC - Use FullyShardedDataParallel (FSDP) when your model cannot fit on one GPU.
# MAGIC
# MAGIC   + See also: Getting Started with FSDP
# MAGIC
# MAGIC - Use Tensor Parallel (TP) and/or Pipeline Parallel (PP) if you reach scaling limitations with FSDP.
# MAGIC
# MAGIC   + Try our Tensor Parallelism Tutorial
# MAGIC
# MAGIC   + See also: TorchTitan end to end example of 3D parallelism
# MAGIC
# MAGIC   ref: https://pytorch.org/tutorials/beginner/dist_overview.html#applying-parallelism-to-scale-your-model

# COMMAND ----------

# MAGIC %md
# MAGIC ### Alternatively, you can try the Microsoft variant of DistributedTorch called "DeepSpeed Distributor"
# MAGIC
# MAGIC DeepSpeed is built on top of Torch Distributor. Though DeepSpeed is more for LLM training, you can try apply this for Image DL models like YOLO
# MAGIC
# MAGIC ref: 
# MAGIC 1. https://github.com/microsoft/DeepSpeed
# MAGIC 1. https://docs.databricks.com/en/machine-learning/train-model/distributed-training/deepspeed.html
# MAGIC 2. https://community.databricks.com/t5/technical-blog/introducing-the-deepspeed-distributor-on-databricks/ba-p/59641

# COMMAND ----------

# MAGIC %md
# MAGIC ### YOLO on Databricks Ref: 
# MAGIC 1. YOLO on databricks, https://benhay.es/posts/object_detection_yolov8/
# MAGIC 2. Distributed Torch + MLflow, https://docs.databricks.com/en/machine-learning/train-model/distributed-training/spark-pytorch-distributor.html
# MAGIC
# MAGIC ![](https://raw.githubusercontent.com/ultralytics/assets/main/im/banner-tasks.png)
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup Environment

# COMMAND ----------

# DBTITLE 1,newest env with .40 ultralytics
# MAGIC %pip install -U ultralytics==8.3.40 opencv-python==4.10.0.84 ray==2.39.0

# COMMAND ----------

# DBTITLE 1,newest env with modified ultralytics package at git repo
# %pip install -U opencv-python==4.10.0.84 ray==2.39.0 git+https://github.com/BlitzBricksterYY-db/ultralytics.git@main

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Load from config
import yaml

# Load the configuration from the config.yaml file
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Extract the DRIFT_DETECTION_VARS
databricks_vars = config.get('databricks_vars', {})

# COMMAND ----------

databricks_host = databricks_vars['databricks_host']
databricks_client_id = databricks_vars['databricks_client_id']
databricks_client_secret_scope = databricks_vars['databricks_client_secret_scope']
databricks_client_secret_key = databricks_vars['databricks_client_secret_key']
databricks_tenant_id = databricks_vars['databricks_tenant_id']
databricks_workspace_id = databricks_vars['databricks_workspace_id']
auth_type = databricks_vars['auth_type']
token_secret_scope = databricks_vars['token_secret_scope']
token_pat_key = databricks_vars['token_pat_key']
catalog_name = databricks_vars['catalog_name']
schema_name = databricks_vars['schema_name']
table_name = databricks_vars['table_name']
project_location = f"/Volumes/{catalog_name}/{schema_name}/{table_name}_vol/"

# COMMAND ----------

# DBTITLE 1,debug switcher
import os

# Enable synchronous execution for debugging,for debugging purpose, show stack trace immediately. Use it in dev mode.
os.environ["CUDA_LAUNCH_BLOCKING"] = "1" 

# Reset to asynchronous execution for production in production
# os.environ["CUDA_LAUNCH_BLOCKING"] = "0" 

# COMMAND ----------

# MAGIC %md
# MAGIC Create your secret scope using "Databricks Secret Vault", for example,
# MAGIC 1. Create your secret scope first for a specific workspace profile: `databricks secrets create-scope {SECRET_SCOPE}`
# MAGIC 2. Put your secret key and value: `databricks secrets put-secret {SECRET_SCOPE} {KEY_NAME}`, here `KEY_NAME` is your key
# MAGIC     - then input the value following the prompt or editor edit/save
# MAGIC 3. (optional) you can also save other key:value pair like databricks_host and workspace_id. `databricks secrets put-secret {SECRET_SCOPE} db_host`
# MAGIC
# MAGIC Ref: https://learn.microsoft.com/en-us/azure/databricks/security/secrets/

# COMMAND ----------

# DBTITLE 1,Set Databricks Host
import os

# Set Databricks workspace URL
try:
    os.environ['DATABRICKS_HOST'] = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get()
    # os.environ['DATABRICKS_WORKSPACE_ID'] = db_wksp_id 
except: 
    os.environ['DATABRICKS_HOST'] = databricks_host
    # os.environ['DATABRICKS_WORKSPACE_ID'] = databricks_vars['workspace_id'] 

# COMMAND ----------

# DBTITLE 1,Set Databricks Token
if databricks_vars['auth_type'] == 'pat':
    try:
        os.environ['DATABRICKS_TOKEN'] = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
    except:
        os.environ['DATABRICKS_TOKEN'] = dbutils.secrets.get(scope=token_secret_scope, key=token_pat_key)

elif databricks_vars['auth_type'] == 'sp':
    # Set Service Principal credentials
    os.environ['DATABRICKS_CLIENT_ID'] = databricks_client_secret_scope
    os.environ['DATABRICKS_CLIENT_SECRET'] = dbutils.secrets.get(scope=databricks_client_secret_scope, key=databricks_client_secret_key)

# COMMAND ----------

import ultralytics
print(ultralytics.__version__)

# COMMAND ----------

import ray
ray.__version__

# COMMAND ----------

# DBTITLE 1,Verify requirements are installed
from ultralytics.utils.checks import check_yolo, check_python, check_latest_pypi_version, check_version, check_requirements

try:
    print("check_yolo", check_yolo())
    print("check_python", check_python())
    print("check_latest_pypi_version", check_latest_pypi_version())
    print("check_version", check_version())
except Exception as e:
    print(f"An error occurred with requirements: {e}") 

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

# Disables the integration with MLflow. 
# By setting the mlflow key to False, you are instructing the ultralytics library not to use MLflow for logging or tracking experiments.
# settings.update({"mlflow":False}) 

# ultralytics level setting with MLflow, if you do want to autolog.
settings.update({"mlflow":True}) 

# # Config MLflow
mlflow.autolog(disable=True)
mlflow.end_run()

# COMMAND ----------

# DBTITLE 1,OpenMP threads
# os.environ["OMP_NUM_THREADS"] = "12"  # if need OpenMP threads

# COMMAND ----------

############################################################################
## Create YOLOC class to capture model results and predict() method ##
############################################################################

class YOLOC(mlflow.pyfunc.PythonModel):
  def __init__(self, point_file):
    self.point_file=point_file

  def load_context(self, context):
    from ultralytics import YOLO
    self.model = YOLO(context.artifacts['best_point'])

  def predict(self, context, model_input):
    return self.model(model_input)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup I/O Path

# COMMAND ----------

spark.sql(f"USE CATALOG {catalog_name}")

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {schema_name}")
spark.sql(f"CREATE TABLE IF NOT EXISTS {schema_name}.{table_name}")
spark.sql(f"CREATE VOLUME IF NOT EXISTS {schema_name}.{table_name}_vol")

# COMMAND ----------

import os
# Config project structure directory
os.makedirs(f'{project_location}/training_runs/', exist_ok=True)
os.makedirs(f'{project_location}/data/', exist_ok=True)
os.makedirs(f'{project_location}/raw_model/', exist_ok=True)
os.makedirs(f'{project_location}/training_results/', exist_ok=True)

# for cache related to ultralytics
os.environ['ULTRALYTICS_CACHE_DIR'] = f'{project_location}/raw_model/'

# or more traditional way, setup folder under DBFS.
# dbfs_project_location = '/dbfs/FileStore/cv_project_location/Nuclei_Instance/'
dbfs_project_location = '/dbfs/tmp/cv_project_location/nuclei_instance/'
os.makedirs(dbfs_project_location, exist_ok=True)

# ephemeral /tmp/ project location on VM
tmp_project_location = "/tmp/training_results/"
os.makedirs(tmp_project_location, exist_ok=True)

# COMMAND ----------

# MAGIC %md
# MAGIC # TODO REMOVE HARDCODED YAML 

# COMMAND ----------

# DBTITLE 1,use this .yaml file under the Volumes
# MAGIC %cat /Volumes/mmt_mlops_demos/cv/data/Nuclei_Instance_Dataset/yolo_dataset/data.yaml

# COMMAND ----------

# DBTITLE 1,define your yaml_path for later training section
yaml_path = "/Volumes/mmt_mlops_demos/cv/data/Nuclei_Instance_Dataset/yolo_dataset/data.yaml"

# COMMAND ----------

# DBTITLE 1,skip this yaml file under workspace
# MAGIC %cat ./data.yaml

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup Working Directory

# COMMAND ----------

os.getcwd()

# COMMAND ----------

dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get().rsplit('/', 1)[0]

# COMMAND ----------

# DBTITLE 1,reset directory to current notebook path
os.chdir('/Workspace/' + dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get().rsplit('/', 1)[0])
os.getcwd()

# COMMAND ----------

# MAGIC %md
# MAGIC ## A few minimal testing examples of Distributed Training 

# COMMAND ----------

# MAGIC %md
# MAGIC ### Tips from Working Mini Example below
# MAGIC Scenaio: If you have 2 minimal and current available workers, each owning 4 GPUs, while the driver node could be CPU or GPU node. The max number could be anything.
# MAGIC
# MAGIC smaller or equaly than 8 (2*4) are supported as we have cluster config as min worker nodes = 2, max = 4. However, as of now, autoscaling is not supported for TorchDistributor on Databricks, so we couldn't go from 8 to 16 GPUs in this example. If you specify something > 8, it will hang there forever without incurring an error msg. 
# MAGIC
# MAGIC Make sure you specify `num_processes` equal to your total # of GPUs over all the available/online nodes, e.g., 2 node * 4 GPUs/node = 8 GPUs = 8 processes.
# MAGIC
# MAGIC Last thing, dont count any GPU from driver, they will not be used for computing.

# COMMAND ----------

# DBTITLE 1,Initial Mini training Example
from pyspark.ml.torch.distributor import TorchDistributor

def train():
    import os
    import torch
    import torch.distributed as dist
    import torch.nn as nn
    import torch.optim as optim
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.utils.data import DataLoader, DistributedSampler, TensorDataset

    # Initialize the process group for distributed training
    backend = 'nccl' if torch.cuda.is_available() else 'gloo'
    dist.init_process_group(backend=backend)
    
    # Set the device based on the local rank
    device_id = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(device_id)
    device = torch.device(f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu')

    # Create the model and move it to the appropriate device
    model = nn.Linear(10, 1).to(device)
    model = DDP(model, device_ids=[device_id])

    # Create dataset and dataloader with distributed sampler
    dataset = TensorDataset(torch.randn(1000, 10), torch.randn(1000, 1))
    sampler = DistributedSampler(dataset)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=32)

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # Training loop
    for epoch in range(10):
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

    # Clean up the process group
    dist.destroy_process_group()
    
try:
    # Instantiate TorchDistributor and run the training function
    distributor = TorchDistributor(num_processes=torch.cuda.device_count(), local_mode=True, use_gpu=True)
    distributor.run(train)
    print("Training completed!")
except Exception as e:
    print(f"Training failed with error {e}")

# COMMAND ----------

# DBTITLE 1,Set Env variables if not already set
try:
    os.environ['DATABRICKS_HOST'] = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get()
    os.environ['DATABRICKS_TOKEN'] = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
    os.environ['DATABRICKS_WORKSPACE_ID'] = databricks_vars['databricks_workspace_id'] 

except: 
    os.environ['DATABRICKS_HOST'] = databricks_host
    os.environ['DATABRICKS_TOKEN'] = dbutils.secrets().get(scope="token_secret_scope", key="pat")
    os.environ['DATABRICKS_WORKSPACE_ID'] = databricks_vars['databricks_workspace_id']

# COMMAND ----------

# DBTITLE 1,A simple multi-gpu I/O test
#: for testing different I/O options
import os
import torch
import torch.distributed as dist
from pyspark.ml.torch.distributor import TorchDistributor

def main():

    # Initialize distributed training
    dist.init_process_group(backend='nccl')  # Or 'gloo' if not using GPUs
    rank = dist.get_rank() # or rank = dist.get_global_rank()
    local_rank = int(os.environ["LOCAL_RANK"])

    print(f"Global rank is: {rank}")
    print(f"Local rank is: {local_rank}")

    # Set the device based on the local rank
    torch.cuda.set_device(local_rank)
    device = torch.device(f'cuda:{local_rank}')

    # output_file_path = 'output.txt'  # this works under local folder in workspace
    output_file_path = '/tmp/output.txt' # this works, but on driver, you cannot see worker nodes' /tmp/ folder.
    # Note: Unity Catalog volumes do not support direct-append or non-sequential (random) writes. For such workloads, you should perform the operations on a local disk first and then copy the results to Unity Catalog volumes. 

    if rank == 0 and os.path.exists(output_file_path):
        os.remove(output_file_path)

    # Only rank 0 should check and create the file
    if rank == 0:
        if not os.path.exists(output_file_path):
            with open(output_file_path, 'w') as f:
                f.write('Hello from rank 0!\n')
            
            print(f"Write from {rank} finished!")

    # Barrier to ensure all processes wait until rank 0 creates the file
    dist.barrier()

    # Other ranks can now access the file
    with open(output_file_path, 'a') as f:
        f.write(f'Hello from rank {rank}!\n')
    print(f"Write from {rank} finished!")

    dist.barrier()

    if rank == 3:
        os.system(f'cp /tmp/output.txt {project_location}/output.txt')
    
    # Clean up the process group
    dist.destroy_process_group()

if __name__ == '__main__':
    # main()
    distributor = TorchDistributor(num_processes=torch.cuda.device_count(), local_mode=True, use_gpu=True)
    distributor.run(main)

# COMMAND ----------

# MAGIC %cat {project_location}/output.txt

# COMMAND ----------

ls -ahlrt /tmp/

# COMMAND ----------

ls -ahlrt /dbfs/tmp/

# COMMAND ----------

cat /dbfs/tmp/output.txt

# COMMAND ----------

ls -ahlrt {project_location}

# COMMAND ----------

# MAGIC %cat {project_location}/output.txt

# COMMAND ----------

# MAGIC %cat {project_location}/output_0.txt

# COMMAND ----------

# MAGIC %cat {project_location}/output_1.txt

# COMMAND ----------

torch.cuda.is_available()

# COMMAND ----------

os.getenv('CUDA_VISIBLE_DEVICES')

# COMMAND ----------

# DBTITLE 1,driver only
# MAGIC %sh
# MAGIC nvidia-smi  -L

# COMMAND ----------

# MAGIC %md
# MAGIC ## Start MLflow Logged Distributed Training

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Setup TensorBoard for monitoring

# COMMAND ----------

# DBTITLE 1,setup tensorboard before running the below distributed training
# MAGIC %load_ext tensorboard
# MAGIC # This sets up our tensorboard settings
# MAGIC # /tmp/training_results/train
# MAGIC # tensorboard --logdir /tmp/training_results/train
# MAGIC experiment_log_dir = f'{tmp_project_location}/train'
# MAGIC %tensorboard --logdir $experiment_log_dir
# MAGIC # This starts Tensorboard

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Setup MLflow

# COMMAND ----------


workspace_path = dbutils.notebook.entry_point.getDbutils().notebook().getContext().tags().get("user").get()
workspace_path = f"/Users/{workspace_path}/"
experiment_name = workspace_path + "MedCellTypes_Instance_Segmentation_Experiment_Managed"
print(f"Setting experiment name to be {experiment_name}")


# COMMAND ----------

os.getcwd()

# COMMAND ----------

mlflow.get_experiment_by_name(experiment_name)

# COMMAND ----------

# DBTITLE 1,mlflow setup for experiment and UC artifact paths
#: mlflow settup
import mlflow

mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks-uc")
mlflow.end_run()
#
# experiment_name = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get().rsplit('/', 1)[0] + "/MedCellTypes_Instance_Segmentation_Experiment_Managed"
# print(f"Setting experiment name to be {experiment_name}")
workspace_path = dbutils.notebook.entry_point.getDbutils().notebook().getContext().tags().get("user").get()
workspace_path = f"/Users/{workspace_path}/"
experiment_name = workspace_path + "MedCellTypes_Instance_Segmentation_Experiment_Managed"
print(f"Setting experiment name to be {experiment_name}")

#: Use UC Volume path to logging MLflow experiment instead of MLflow-managed artifact storage: dbfs:/databricks/mlflow-tracking/<experiment-id>.
# project_location = '/Volumes/yyang/computer_vision/Nuclei_Instance/'
ARTIFACT_PATH = f"dbfs:{project_location}artifact"
print(f"Creating experiment ARTIFACT_PATH to be {ARTIFACT_PATH}")
if mlflow.get_experiment_by_name(experiment_name) is None:
    mlflow.create_experiment(name=experiment_name, artifact_location=ARTIFACT_PATH)

# COMMAND ----------

# MAGIC %md
# MAGIC ### SingleNode-MultiGPU version

# COMMAND ----------

# DBTITLE 1,SingleNode-MultiGPU version
from pyspark.ml.torch.distributor import TorchDistributor

settings.update({"mlflow":True}) # if you do want to autolog.
mlflow.autolog(disable = False)


def train_fn(device_list = [0,1,2,3], yaml_path = None, active_run_id = None):

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
            data=yaml_path, # put your .yaml file path here.
            epochs=50,
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
            mlflow.log_artifact(yaml_path, "input_data_yaml")
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

    #: use this experiment_name
    mlflow.set_experiment(experiment_name)
    experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
    # Reset MLFLOW_RUN_ID, so we dont bump into the wrong one.
    if 'MLFLOW_RUN_ID' in os.environ:
        del os.environ['MLFLOW_RUN_ID']

    #: setting global env
    # Set Databricks workspace URL and token
    os.environ['DATABRICKS_HOST'] = db_host = 'https://adb-984752964297111.11.azuredatabricks.net'
    os.environ['DATABRICKS_WORKSPACE_ID'] = db_wksp_id = '984752964297111'
    os.environ['DATABRICKS_TOKEN'] = db_token = dbutils.secrets.get(scope="yyang_secret_scope", key="pat")

    print(os.environ['DATABRICKS_HOST'])
    print(os.environ['DATABRICKS_WORKSPACE_ID'])
    print(os.environ['DATABRICKS_TOKEN']) # anything from vault would be redacted print.
    #  
    os.environ['MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING'] = "true"
    print(f"MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING set to {os.environ['MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING']}")
    os.environ['MLFLOW_EXPERIMENT_NAME'] = experiment_name
    print(f"MLFLOW_EXPERIMENT_NAME set to {os.environ['MLFLOW_EXPERIMENT_NAME']}")
    os.environ['DATABRICKS_WORKSPACE_ID'] = db_wksp_id  # Set the workspace ID
    print(f"DATABRICKS_WORKSPACE_ID set to {os.environ['DATABRICKS_WORKSPACE_ID']}")


    #: caculate # of GPUs
    num_gpus = torch.cuda.device_count()
    device_list = list(range(num_gpus))
    print("num_gpus:", num_gpus)
    print("device_list:", device_list)

    yaml_path = yaml_path # reflect your dataset yolo format .yaml path.

    with mlflow.start_run(experiment_id=experiment_id) as run:
        active_run_id = mlflow.last_active_run().info.run_id
        active_run_name = mlflow.last_active_run().info.run_name

        print("For master triggering run, active_run_id is: ", active_run_id)
        print("For master triggering run, active_run_name is: ", active_run_name)
        print(f"For master triggering run, active_run_id is: '{active_run_id}' and active_run_name is: '{active_run_name}'.")
        print(f"All worker runs will be logged into the same run id '{active_run_id}' and name '{active_run_name}'.")

        distributor = TorchDistributor(num_processes=num_gpus, local_mode=True, use_gpu=True)      
        distributor.run(train_fn, device_list = device_list, yaml_path = yaml_path, active_run_id = active_run_id)


# Best Practice You Are! Previously we have 2 runs, one for master and the other for all workers.
# 1. master run id for recording system metrics only
# 2. Inner run id for auto-logging and manually logging other artifacts.

# Now I have changed the code to merge outter run and inner run to be the same run id. less confusion.

# COMMAND ----------

# DBTITLE 1,With Default PyTorch Flavor SingleNode-MultiGPU version
# from pyspark.ml.torch.distributor import TorchDistributor

# settings.update({"mlflow":True}) # if you do want to autolog.
# mlflow.autolog(disable = False)


# def train_fn(device_list = [0,1,2,3], yaml_path = None, active_run_id = None):

#     from ultralytics import YOLO
#     import torch
#     import mlflow
#     import torch.distributed as dist
#     from ultralytics import settings
#     from mlflow.types.schema import Schema, ColSpec
#     from mlflow.models.signature import ModelSignature

#     ############################

#     ##### Setting up MLflow ####
#     # We need to do this so that different processes that will be able to find mlflow
#     os.environ['DATABRICKS_HOST'] = db_host # pending replace with db vault secret
#     os.environ['DATABRICKS_TOKEN'] = db_token # pending replace with db vault secret 
#     os.environ['MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING'] = "true"
#     os.environ['MLFLOW_EXPERIMENT_NAME'] = experiment_name
#     os.environ['DATABRICKS_WORKSPACE_ID'] = db_wksp_id  # Set the workspace ID
#     # We set the experiment details here
#     experiment = mlflow.set_experiment(experiment_name)
    
#     #
#     with mlflow.start_run(run_id=active_run_id) as run:
#         model = YOLO(f"{project_location}/raw_model/yolov8n.pt")
#         model.train(
#             batch=8,
#             device=device_list,
#             data=yaml_path, # put your .yaml file path here.
#             epochs=3,
#             project=f'{tmp_project_location}',
#             exist_ok=True,
#             fliplr=1,
#             flipud=1,
#             perspective=0.001,
#             degrees=.45
#         )

#     # active_run_id = mlflow.last_active_run().info.run_id
#     # print("For YOLO autologging, active_run_id is: ", active_run_id)

#     # # after training is done.
#     # if not dist.is_initialized():
#     #   # import torch.distributed as dist
#     #   dist.init_process_group("nccl")

#     local_rank = int(os.environ["LOCAL_RANK"])
#     global_rank = int(os.environ["RANK"])
    
#     if global_rank == 0:

#         active_run_id = mlflow.last_active_run().info.run_id
#         print("For YOLO autologging, active_run_id is: ", active_run_id)

#         # Get the list of runs in the experiment
#         runs = mlflow.search_runs(experiment_names=[experiment_name], order_by=["start_time DESC"], max_results=1)

#         # Extract the latest run_id
#         if not runs.empty:
#             latest_run_id = runs.iloc[0].run_id
#             print(f"Latest run_id: {latest_run_id}")
#         else:
#             print("No runs found in the experiment.")


#         with mlflow.start_run(run_id=latest_run_id) as run:
#             mlflow.log_artifact(yaml_path, "input_data_yaml")
#             mlflow.log_params({"rank":global_rank})
#             mlflow.pytorch.log_model(model.trainer.best, "model", signature=signature)
#             # mlflow.pytorch.log_model(YOLO(str(model.trainer.best)), "model", signature=signature)

#             # yolo_wrapper = YOLOC(model.trainer.best)
#             # mlflow.pyfunc.log_model(
#             #     artifact_path="model",
#             #     artifacts={'model_path': str(model.trainer.save_dir), "best_point": str(model.trainer.best)},
#             #     python_model=yolo_wrapper,
#             #     signature=signature)

#     # clean up
#     if dist.is_initialized():
#         dist.destroy_process_group()

#     return "finished" # can return any picklable object


# if __name__ == "__main__":

#     #: use this experiment_name
#     mlflow.set_experiment(experiment_name)
#     experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
#     # Reset MLFLOW_RUN_ID, so we dont bump into the wrong one.
#     if 'MLFLOW_RUN_ID' in os.environ:
#         del os.environ['MLFLOW_RUN_ID']

#     #: setting global env
#     # Set Databricks workspace URL and token
#     os.environ['DATABRICKS_HOST'] = db_host = 'https://adb-984752964297111.11.azuredatabricks.net'
#     os.environ['DATABRICKS_WORKSPACE_ID'] = db_wksp_id = '984752964297111'
#     os.environ['DATABRICKS_TOKEN'] = db_token = dbutils.secrets.get(scope="yyang_secret_scope", key="pat")

#     print(os.environ['DATABRICKS_HOST'])
#     print(os.environ['DATABRICKS_WORKSPACE_ID'])
#     print(os.environ['DATABRICKS_TOKEN']) # anything from vault would be redacted print.
#     #  
#     os.environ['MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING'] = "true"
#     print(f"MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING set to {os.environ['MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING']}")
#     os.environ['MLFLOW_EXPERIMENT_NAME'] = experiment_name
#     print(f"MLFLOW_EXPERIMENT_NAME set to {os.environ['MLFLOW_EXPERIMENT_NAME']}")
#     os.environ['DATABRICKS_WORKSPACE_ID'] = db_wksp_id  # Set the workspace ID
#     print(f"DATABRICKS_WORKSPACE_ID set to {os.environ['DATABRICKS_WORKSPACE_ID']}")


#     #: caculate # of GPUs
#     num_gpus = torch.cuda.device_count()
#     device_list = list(range(num_gpus))
#     print("num_gpus:", num_gpus)
#     print("device_list:", device_list)

#     yaml_path = yaml_path # reflect your dataset yolo format .yaml path.

#     with mlflow.start_run(experiment_id=experiment_id) as run:
#         active_run_id = mlflow.last_active_run().info.run_id
#         active_run_name = mlflow.last_active_run().info.run_name

#         print("For master triggering run, active_run_id is: ", active_run_id)
#         print("For master triggering run, active_run_name is: ", active_run_name)
#         print(f"For master triggering run, active_run_id is: '{active_run_id}' and active_run_name is: '{active_run_name}'.")
#         print(f"All worker runs will be logged into the same run id '{active_run_id}' and name '{active_run_name}'.")

#         distributor = TorchDistributor(num_processes=num_gpus, local_mode=True, use_gpu=True)      
#         distributor.run(train_fn, device_list = device_list, yaml_path = yaml_path, active_run_id = active_run_id)


# # Best Practice You Are! Previously we have 2 runs, one for master and the other for all workers.
# # 1. master run id for recording system metrics only
# # 2. Inner run id for auto-logging and manually logging other artifacts.

# # Now I have changed the code to merge outter run and inner run to be the same run id. less confusion.

# COMMAND ----------

#: use this experiment_name
mlflow.set_experiment(experiment_name)
experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
# Reset MLFLOW_RUN_ID, so we dont bump into the wrong one.
if 'MLFLOW_RUN_ID' in os.environ:
    del os.environ['MLFLOW_RUN_ID']

# after training is done.
if not dist.is_initialized():
  # import torch.distributed as dist
  dist.init_process_group("nccl")
model = YOLO(f"{project_location}/raw_model/yolov8n.pt")
model.train(
    batch=8,
    device=device_list,
    data=yaml_path, # put your .yaml file path here.
    epochs=3,
    project=f'{tmp_project_location}',
    exist_ok=True,
    fliplr=1,
    flipud=1,
    perspective=0.001,
    degrees=.45
)

# clean up
if dist.is_initialized():
    dist.destroy_process_group()

# COMMAND ----------

#  mlflow.pytorch.log_model(model.trainer.best, "model", signature=signature) # this will fail
mlflow.pytorch.log_model(YOLO(str(model.trainer.best)), "model", signature=signature) # this succeeded

# COMMAND ----------

mlflow.log_artifact(yaml_path, "input_data_yaml")
mlflow.log_params({"rank":0})

# COMMAND ----------

mlflow.end_run()
