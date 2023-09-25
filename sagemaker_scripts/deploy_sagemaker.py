import argparse
from datetime import datetime
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Union
from botocore.client import BaseClient as Boto3Client
import boto3
import botocore.client
import yaml
from dotenv import load_dotenv
from sagemaker import get_execution_role

logging.basicConfig(level=logging.INFO)  # Set logging level to INFO

load_dotenv()


def check_model_name(model_name: str, models_list: List[Dict[str, Union[str, datetime]]]) -> bool:
    """
    Check if a model with the specified name exists in a given list of models.

    Parameters:
        model_name (str): The name of the model to check.
        models_list (List[Dict[str, Union[str, datetime]]]): The list of models to search.

    Returns:
        bool: True if the model exists, False otherwise.
    """
    for model in models_list:
        if model["ModelName"] == model_name:
            return True
    return False


def get_model_name(model_uri: str) -> str:
    """
    Extract the model name from the model URI.

    Parameters:
        model_uri (str): The URI of the model.

    Returns:
        str: The name of the model.
    """
    parent_name_1 = Path(model_uri).parents[1].name
    if "tensorflow-training" in parent_name_1:
        return parent_name_1
    else:
        return Path(model_uri).parent.name


def create_model(
    model_uri: str,
    model_name: str,
    triton_image_uri: str,
    sagemaker_triton_default_model_name: str,
    role: str,
    sm_client: Boto3Client,
) -> bool:
    """
    Create a model in SageMaker.

    Parameters:
        model_uri (str): The URI of the model.
        model_name (str): The name for the model.
        triton_image_uri (str): The URI of the Triton image.
        sagemaker_triton_default_model_name (str): Must match the X in `train.py --model_name X` which is the root folder name in artifact zip file
        role (str): The ARN of the IAM role.
        sm_client (botocore.client.BaseClient): The SageMaker client.

    Returns:
        bool: True if the model was created successfully, False otherwise.
    """
    container = {
        "Image": triton_image_uri,
        "ModelDataUrl": model_uri,
        "Environment": {"sagemaker_triton_default_model_name": sagemaker_triton_default_model_name},
    }

    is_model_found = check_model_name(model_name, sm_client.list_models()["Models"])
    if is_model_found:
        logging.warning(
            f"Model {model_name} already exist. If you wish to overwrite, please manually remove it through sagemaker UI."
        )
        return False
    try:
        create_model_response = sm_client.create_model(
            ModelName=model_name, ExecutionRoleArn=role, PrimaryContainer=container
        )
        logging.info("Model Arn: " + create_model_response["ModelArn"])
        return True
    except Exception as e:
        logging.error(f"Failed at creating model. {e}")
        raise e


def check_endpoint_initialized(endpoint_name: str, sm_client: Boto3Client):
    """
    Check if a SageMaker endpoint is initialized.

    Parameters:
        endpoint_name (str): The name of the endpoint.
        sm_client (botocore.client.BaseClient): The SageMaker client.
    """

    ### Takes ~10 minutes to download and deploy triton server docker image on AWS
    resp = sm_client.describe_endpoint(EndpointName=endpoint_name)
    status = resp["EndpointStatus"]
    logging.info("Status: " + status)

    while status in ["Creating", "Updating"]:
        time.sleep(60)
        resp = sm_client.describe_endpoint(EndpointName=endpoint_name)
        status = resp["EndpointStatus"]
        logging.info("Status: " + status)

    logging.info("Arn: " + resp["EndpointArn"])
    logging.info("Status: " + status)

    if status != "InService":
        raise Exception("Failed to initialize the endpoint. Please check logs.")


def create_endpoint(
    endpoint_name: str,
    model_name: str,
    sm_client: botocore.client.BaseClient,
    instance_type: str = "ml.t2.medium",
    instance_count: int = 1,
):
    """
    Create or update a SageMaker endpoint.

    Parameters:
        endpoint_name (str): The name of the endpoint.
        model_name (str): The name of the model.
        sm_client (botocore.client.BaseClient): The SageMaker client.
        instance_type (str, optional): The type of EC2 instance for the endpoint. Defaults to "ml.t2.medium".
        instance_count (int, optional): The number of instances for the endpoint. Defaults to 1.
    """
    endpoint_config_name = endpoint_name + "-" + datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    create_endpoint_config_response = sm_client.create_endpoint_config(
        EndpointConfigName=endpoint_config_name,
        ProductionVariants=[
            {
                "InstanceType": instance_type,
                "InitialVariantWeight": 1,
                "InitialInstanceCount": instance_count,
                "ModelName": model_name,
                "VariantName": "AllTraffic",
            }
        ],
    )

    try:
        create_endpoint_response = sm_client.create_endpoint(
            EndpointName=endpoint_name, EndpointConfigName=endpoint_config_name
        )
    except:
        create_endpoint_response = sm_client.update_endpoint(
            EndpointName=endpoint_name, EndpointConfigName=endpoint_config_name
        )
    print("Endpoint Arn: " + create_endpoint_response["EndpointArn"])


def main(config):
    # Access values from config dictionary
    model_uri = config["model_uri"]
    triton_image_uri = config["triton_image_uri"]
    role = os.getenv("SM_ARN_ROLE") or get_execution_role()  #
    env = os.getenv("ENV")
    endpoint_name = config["endpoint_name"]
    endpoint_name = f"{endpoint_name}-{env}"  # add prod/stage to the endpoint for CD in PRs
    instance_type = config["instance_type"]
    instance_count = config["instance_count"]
    sagemaker_triton_default_model_name = config["sagemaker_triton_default_model_name"]

    sm_client = boto3.client(service_name="sagemaker")

    model_name = get_model_name(model_uri)
    create_model(model_uri, model_name, triton_image_uri, sagemaker_triton_default_model_name, role, sm_client)
    create_endpoint(endpoint_name, model_name, sm_client, instance_type, instance_count)
    check_endpoint_initialized(endpoint_name, sm_client)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deploy trained models.")
    parser.add_argument("--config", required=True, help="Path to the configuration YAML file.")

    args = parser.parse_args()

    # Read the YAML file
    with open(args.config, "r") as file:
        config = yaml.safe_load(file)

    main(config)
