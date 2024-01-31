# Digit Classification Training and Deployment

![tests](https://github.com/jahaniam/ml-training-pipeline/actions/workflows/test.yml/badge.svg)
![deploy](https://github.com/jahaniam/ml-training-pipeline/actions/workflows/deploy-prod.yml/badge.svg)



## About
This project implements a Machine Learning (ML) pipeline with Continuous Integration and Continuous Deployment (CI/CD) capabilities. It supports large-scale training on the cloud as well as local training and deployment.


### Tech Stack

- **Large Scale Training:** AWS sagemaker

- **Inference Server:** Nvidia Triton - supports most frameworks. It has a scheduler, dynamic batching, and concurrency for model optimization. it supports gRPC for model inference on edge for streaming data.

- **Experiment Tracking:** Weights and Biases (https://wandb.ai/mnist_prototype/digit_classification/)

- **ML Framework:** Tensorflow + Keras - currently Keras is natively supported in TensorFlow. [New release](https://keras.io/keras_core/#:~:text=Keras%20Core%20makes%20it%20possible,%2C%20JAX%2C%20and%20PyTorch%20workflows) of Keras will support Pytorch, Jax, and Tensorflow. It will become a versatile framework in production and research.

- **CI/CD:** Github Actions

- **UI:** Streamlit (https://simple-ml-pipeline.streamlit.app/)

Other notable tools and third-party solutions include Ray (distributed training), Neptune AI(experiment tracking and model registry), etc.


## CI/CD Pipelines

- *test.yml:* Triggered on pull requests, runs unit tests for each module, and ensures compliance with formatting and pre-commit settings.

- *deploy-stage.yml:* Triggered on pull requests, monitors model changes and, if detected, deploys the new changes to the staging endpoint.


- *deploy-prod:* Triggered on pushes to the main branch, monitors model changes and, if detected, deploys the new changes to the production endpoint.



## Code Structure

- *.github/workflows:* Contains GitHub Actions configurations (see the `CI/CD Pipelines` section).

- *digit_classification:* Contains project code for training the model (`train.py`).

- *sagemaker_scripts:* Scripts for deployment and training, both large-scale and local. Although `train.py` from `digit_classification` can be used, [`train_sagemaker_hyperparam.ipynb`](https://github.com/jahaniam/ml-pipeline/blob/main/sagemaker_scripts/train_sagemaker_hyperparam.ipynb) is preferred as it generates artifacts that match the deployment model artifacts.
  - `models/digit_classification.yaml`: Contains model configuration for production/staging.

- *streamlit_app:* A simple user interface to test the available production and staging endpoints.

## Training
#### Setup
Within `sagemaker_scripts/`, create a `.env` file and populate it with your keys:

```
SM_ARN_ROLE=<YOUR_AWS_SAGEMAKER_ARN>
WANDB_API_KEY=<YOUR_WEIGHTS_AND_BIASES_API_KEY>
```
Ensure your AWS CLI is set up: aws configure.
TODO: Pass `aws_secret_access_key` and `aws_access_key_id` environments to the training container.
### Train
Initiate a Jupyter Lab instance within a container using:
`docker compose up --build`
A Jupyter Notebook server is then accessible at `127.0.0.1:8888` where you can open [`sagemaker_scripts/train_sagemaker_hyperparam.ipynb`](https://github.com/jahaniam/ml-pipeline/blob/main/sagemaker_scripts/train_sagemaker_hyperparam.ipynb) .


 - Additionally, a vscode remote connection can be utilized to run the code.

##### Hyperparameter tuning
```
hyperparameter_ranges  = {
"lr": CategoricalParameter([0.0001, 0.001, 0.01]),
"batch_size": CategoricalParameter([128, 256, 512]),}
```
Hyperparameters tuning is set up for `learning rate` and `batch_size`. Additional parameters can easily be added to the above.


```
tuner  =  HyperparameterTuner(..., max_jobs=6,max_parallel_jobs=1,
early_stopping_type="Off",autotune=False, ...)
```
Adjusting these parameters, such as increasing the number of jobb, maximum parallel jobs, or changing the `instance_type` to a GPU instance, can expedite training.

#### Experiment Tracking
Once the models are trained, we can see the experiments and metrics here https://wandb.ai/mnist_prototype/digit_classification/
We can also see the training jobs on Sagemaker panel.
Post training, experiment outcomes and metrics can be viewed in [Weights and Biases Project Dashboard](https://wandb.ai/mnist_prototype/digit_classification/) here and within the SageMaker panel.

## Deployment

#### Cloud
Upon successful training, an URI to the model on S3 is provided, e.g.,
`s3://sagemaker-us-east-1-633875729936/digit_classification/models/tensorflow-training-230925-2120-001-413e8660/output/model.tar.gz` . While using a model registry is advisable, this demo utilizes this path directly.

Update
`sagemaker_scripts/models/digit_classification.yaml`:


```
endpoint_name: digit-classification
model_uri: s3://sagemaker-us-east-1-633875729936/digit_classification/models/tensorflow-training-2023-09-25-05-55-53-638/model.tar.gz
triton_image_uri: 785573368785.dkr.ecr.us-east-1.amazonaws.com/sagemaker-tritonserver:21.08-py3
instance_type: ml.c5.large
instance_count: 1
sagemaker_triton_default_model_name: mnist
```
To make the downtime as low as possible we can increase the instance count.
We can use higher end instance type to increase the speed. (Speed could be improved if we quantize the model to INT8)
Amend the URI with the new path obtained post training and submit a pull request. GitHub Actions will deploy the PR model to the staging environment. Upon successful staging deployment, merge the PR to the main branch where deploy-prod.yml workflow will promote the model to production.
#### Local
Please follow [`sagemaker_scripts/train_sagemaker_hyperparam.ipynb`](https://github.com/jahaniam/ml-pipeline/blob/main/sagemaker_scripts/train_sagemaker_hyperparam.ipynb) for local deployment. Given a model artifact, an Nvidia Triton Server docker container is launched locally that utilizes local GPU. The current model can run `1070.22 infer/sec, latency 933 usec`

## Improvements:
- Incorporate data augmentation within the dataloader to mitigate overfitting and enhance model generalization.
- For large datasets, employ batch loading instead of loading the entire dataset into RAM.
- Utilize [Quantized Aware Training](https://www.tensorflow.org/model_optimization/guide/quantization/training) to train the model with reduced precision (FP16 or INT8). This requires rewriting the model in a Functional format as Model Subclassing is not supported.

- Convert the model to TensorRT, quantize the trained model for improved speed, and deploy FP16 or INT8 quantized models. Given that the inference engine is Triton Server, integrating TensorRT can be achieved with minimal adjustments.
