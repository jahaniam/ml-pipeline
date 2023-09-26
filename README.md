# Digit Classification Training and Deployment

![tests](https://github.com/jahaniam/ml-training-pipeline/actions/workflows/test.yml/badge.svg)

![deploy](https://github.com/jahaniam/ml-training-pipeline/actions/workflows/deploy-prod.yml/badge.svg)



## About


This projects implements an ML pipeline with CI/CD. It supports both large scale training on the cloud and also local training and deployment.

Tech Stack:

- **Large Scale Training:** AWS sagemaker

- **Inference Server:** Nvidia Triton

- **Experiment Tracking:** Weights and Biases

- **ML Framework:** Tensorflow Keras

- **CI/CD:** Github Actions

- **UI:** Streamlit

There are many other tools and third party companies which have their own advantages e.g. Ray, Neptune AI, Pytorch, etc.


## CI/CD Pipelines

- **test.yml:** on PR, runs the unit test for each module and also makes sure the format and pre-commit settings are applid.

- **deploy-stage.yml:** on PR, it monitors the changes to the model and if they changed, it deploys the new changes to the staging endpoint

- **deploy-prod:** on push to the main, it monitors the changes to the model and if they changed, it deploys the new changes to the production endpoint



## Code Structure

- .**github/workflows:** includes codes for github actions. see `CI/CD Pipelines` section

- **digit_classification:** project code developed to train the model (train.py)

- **sagemaker_scripts:** scripts for deployment and also large scale and local training. Although `train.py` from `digit_classification` could be used, `train_sagemaker_hyperparam.ipynb` notebook is prefered becaus the artifacts it creates matches the deployment model artifacts.
	- **models/digit_classification.yaml:** model configuration in production/staging.

- **streamlit_app:** a simple user interface which can test the production and staging endpoints available.

## Setup
Before starting training you will need to set up few accoutns:
get your AWS_

## Training
### Setup
in `sagemaker_scrips/` create a `.env` folder and fill it with your keys.:
```
SM_ARN_ROLE=<YOUR_AWS_SAGEMAKER_ARN>
WANDB_API_KEY=<YOUR_WEIGHTS_AND_BIASES_API_KEY>
```
Make sure your aws cli is also setup: `aws configure`
TODO: Pass `aws_secret_access_key` and `aws_access_key_id` enviroments to the training container
### Train
For training, we bring up a jupyter lab in container.
`docker compose up --build`
a jupyter notebook server is available at `127.0.0.1:8888` and you can open the `sagemaker_scrips/train_sagemaker_hyperparam.ipynb`
 - You can use `vscode`remote connection and run the code as well.

##### Hyperparameter tuning
```
hyperparameter_ranges  = {
"lr": CategoricalParameter([0.0001, 0.001, 0.01]),
"batch_size": CategoricalParameter([128, 256, 512]),}
```
Hyperparameters tuning are only setup on `learning rate` and `batch_size` . We can easiy add them in the
```
tuner  =  HyperparameterTuner(..., max_jobs=6,max_parallel_jobs=1,
early_stopping_type="Off",autotune=False, ...)
```
We can play with these parameters and increase the number of jobs and also changing the `instance_type` to be a gpu instance for faster training.
#### Experiment Tracking
Once the models are trained, we can see the experiments and metrics here https://wandb.ai/mnist_prototype/digit_classification/
We can also see the training jobs on Sagemaker panel.

### Deployment
Once the network is trained, the output is a uri to the model on s3 e.g.
`s3://sagemaker-us-east-1-633875729936/digit_classification/models/tensorflow-training-230925-2120-001-413e8660/output/model.tar.gz` . In general it is better to use model registry but for this demo we directly use this path.

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
Update the uri with the new path we get from training and submit a PR. Github actions will deploy the PR model on staging enviroment and if it is successful it can be merged to main where it will be deployed to prod by `deploy-prod.yml `workflow.

**Improvements:**
- Augmentation could be added to the dataloader to reduce overfitting and increasae the generalization.
- Since the mnist dataset size is low, all the dataset is loaded in RAM. For big datasets, it should load batch wise.
- Model is trained on FP32. We can use train using [Quantized Aware Training](https://www.tensorflow.org/model_optimization/guide/quantization/training).  Since `Model Subclassing` is not supported, model needs to be rewritten in Functional format.
> -   Model building:  [`tf.keras`](https://www.tensorflow.org/api_docs/python/tf/keras)  with only Sequential and Functional models.
- We can convert the model to TensorRT and quantize the trained model for speed and use `FP16` or `INT8` quantized models and convert to TensorRT. Since the inference engine is Triton Server, it supports TensorRT and it can be added with minimal change.
