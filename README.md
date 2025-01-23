# E2E Computer Vision with YOLO on Databricks for HIMSS

## Table of Contents
- [Overview](#overview)
- [Key Features](#key-features)
- [Use Cases](#use-cases)
- [Getting Started](#getting-started)
- [Dataset](#dataset)
- [Contributing](#contributing)
- [License](#license)

## Overview
This solution empowers users to efficiently execute computer vision projects using the YOLO framework on Databricks. It offers scalable configurations, streamlines the machine learning lifecycle, and provides end-to-end experiment tracking with MLflow.

## Key Features
- **Efficient Setup**: Minimal setup effort with seamless integration into MLflow and Unity Catalog for well-organized machine learning lifecycle management.
- **Scalable Configurations**: Single-node/single-GPU to multi-node/multi-GPU setups.
- **Data Augmentation and Fine-Tuning**: Includes reusable code for data augmentation and model fine-tuning, enabling flexibility for diverse use cases.
- **Integration with MLflow**: Experiment tracking and logging of all metrics and parameters.
- **Comprehensive Monitoring**: Detailed monitoring, comparison, and evaluation of experiments.
- **Cloud-Native Architecture**: Access to abundant compute resources and cost-effective scaling.

## Use Cases
- Accelerate time-to-value for computer vision projects by simplifying scalable deployments.
- Simplify machine learning lifecycle management through MLflow and Unity Catalog integration.
- Run YOLO models across single-node and multi-node setups, reducing bottlenecks in experimentation and deployment.
- Optimize training time for large datasets with multi-node, multi-GPU setups.

## Getting Started
To get started with this solution, follow these steps:
1. Set up your Databricks environment with the required dependencies.
2. Clone this repository and navigate to the root directory.
3. Follow the instructions in the setup directory to configure your environment.
4. Run the example notebooks to get started with the YOLO framework.

### Multi-Node Multi-GPU Setup on Databricks
To set up a multi-node, multi-GPU environment on Databricks, follow these steps:

1. Ensure you have access to an Azure Databricks Workspace with a Premium Pricing Tier for seamless integration with the Microsoft Fabric Lakehouse.
2. Create a cluster with the following configuration:
   - **Cluster Type**: Multi-node cluster, single-user.
   - **Databricks Runtime Version**: At least 13.0 ML (GPU, Scala 2.12, Spark 3.4.0).
   - **Worker Type**: Standard_NC4as_T4_v3 with at least 2 workers (recommended 8 workers).
   - **Driver Type**: Same as worker type.
   - **Autoscaling**: Disabled.
3. Ensure you have enough CPU cores of the specified type. If not, request a quota increase from your Azure subscription administrator.
4. Install a cluster-scoped init script in your cluster. Use the `env_update.sh` script for this purpose.

## Dependencies
To install the required dependencies, run the following command at the top of the relevant notebooks:
```bash
%pip install -U ultralytics==8.3.40 opencv-python==4.10.0.84 ray==2.39.0
``` 

## Dataset
This solution is designed to handle datasets like HLS brain cancer imaging. Please contact us for more information on using this solution with your specific dataset.

## Contributing
This is a private repository for internal use only at this time. We welcome contributions to this repository. Please submit a pull request with your changes and a brief description of your contribution.

## License
This repository is licensed under the [Apache License, Version 2.0, January 2004](http://www.apache.org/licenses/). By using this repository, you agree to the terms and conditions of the license.