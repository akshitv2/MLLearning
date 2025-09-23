---
title: SageMaker
nav_order: 13
parent: Notes
layout: default
---

# SageMaker![img_1.png](img_1.png)

Amazon SageMaker is a fully managed service from AWS that helps developers and data scientists build, train, and deploy
machine learning (ML) models quickly and easily.
covers the entire ML lifecycle, from data labeling to model deployment and monitoring.

## Key Components

SageMaker is not a single tool but a collection of services that work together. Some of the key components are:

1. **Sagemaker Studio**:
    - Integrated development environment (IDE) for ML
    - Provides a single web-based interface where you can perform all the steps of the ML workflow
2. **SageMaker Notebook Instances**:
    - These are managed Jupyter notebooks that come pre-configured with popular ML frameworks like TensorFlow, PyTorch,
      and scikit-learn.
    - Similar to colab but can launch powerful machine with deep AWS integration
    - 🟢 Great for exploratory data analysis and developing your model code
3. **SageMaker Training**:
    - Allows you to train your models at scale.
    - SageMaker handles the heavy lifting of managing the compute instances,distributing the training workload, and
      saving the final model artifact.
    - Can easily train a model on massive datasets without needing to manage a single server.
4. **SageMaker Inference (Deployment)**:
    - To deploy it to a production environment.
    - Creates an endpoint where your application can send data to get predictions.
    - ℹ️ AWS Endpoint is not a public API but an endpoint you call using AWS SDK
    - SageMaker inference can actually be done in multiple ways:
        - Real time inference: Using an always "on" endpoint
        - Batch Transform: Run on dataset on s3 and write to s3
        - Serverless Endpoints: On demand
5. **SageMaker Data Wrangler**:
    - Tool for data preparation and feature engineering.
    - Helps you aggregate and prepare data from various sources and transform it for training.
6. **SageMaker Pipelines**:
    - Allows you to automate and manage the steps of your ML workflow

### Example Calling sagemaker endpoint

```python
import boto3
import json

runtime = boto3.client('sagemaker-runtime')

data = {"features": [5.1, 3.5, 1.4, 0.2]}  # Example for Iris dataset
payload = json.dumps(data)

response = runtime.invoke_endpoint(
    EndpointName='my-endpoint',
    ContentType='application/json',
    Body=payload
)

result = json.loads(response['Body'].read())
print(result)
```

<hr>

## The Machine Learning Lifecycle

Systematic, iterative process that guides the development and deployment of ML models.
The ML lifecycle can be broken down into several key stages:

1. Data Preparation:
    - Often the most time-consuming part of the process.
    - Involves collecting, cleaning, and transforming raw data into a format suitable for training
    - Tasks include handling missing values, removing duplicates, feature engineering, and splitting the data into
      training, validation, and test sets.
    - 💡Sagemaker Data Wrangler
2. Model Building:
    - Choose a suitable algorithm or model architecture for your problem.
    - You define the model structure, including the number of layers (for a neural network) or the type of algorithm (
      like a decision tree or a linear regression model).
    - 💡 Sagemaker Studio Notebooks
3. Training:
    - Feed the training data to the model to minimize the error or loss.
    - This process is computationally intensive and often requires powerful hardware.
    - 💡 Sagemaker Training
4. Deployment:
    - After the model is trained and validated, needs to be made available for use.
    - Involves packaging the model and deploying it to a production environment where it can receive new data and
      provide predictions.
    - This is also known as inference.
    - 💡 SageMaker Inference
5. Monitoring:
    - Crucial to monitor its performance over time.
    - Involves tracking metrics like prediction accuracy, latency, and resource usage.
    - Models can degrade over time due to changes in data distribution, a phenomenon known as model drift. Monitoring
      helps you detect and address these issues.
    - 💡SageMaker Model Monitor

## Getting Started with SageMaker
### Setting up an AWS Account and IAM Roles:
   - Create an IAM User, Attach IAM policies: `AmazonSageMakerFullAccess` needed for broad acess 
   - Create an Execution Role: SageMaker needs an IAM execution role to perform actions on your behalf (like S3 access)
