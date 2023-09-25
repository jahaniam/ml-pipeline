import argparse
import datetime
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Union

import boto3
import botocore.client
import yaml
from dotenv import load_dotenv
from sagemaker import get_execution_role

logging.basicConfig(level=logging.DEBUG)  # Set logging level to INFO

load_dotenv()


def check_model_name(model_name: str, models_list: List[Dict[str, Union[str, datetime.datetime]]]) -> bool:
    for model in models_list:
        if model["ModelName"] == model_name:
            return True
    return False


def get_model_name(model_uri: str) -> str:
    return Path(model_uri).parents[1].name


def create_model(
    model_uri: str,
    model_name: str,
    triton_image_uri: str,
    role: str,
    sm_client: botocore.client.BaseClient,
) -> bool:
    container = {
        "Image": triton_image_uri,
        "ModelDataUrl": model_uri,
        "Environment": {"SAGEMAKER_TRITON_DEFAULT_MODEL_NAME": "mnist"},
    }

    is_model_found = check_model_name(model_name, sm_client.list_models()["Models"])
    if is_model_found:
        logging.warning(
            f"Skipping deployment. Model {model_name} is already exist. If you wish to overwrite, please manually remove it through sagemaker UI."
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


def check_endpoint_initialized(endpoint_name: str, sm_client: botocore.client.BaseClien):
    ### Takes ~10 minutes to download and deploy triton server docker image on AWS
    resp = sm_client.describe_endpoint(EndpointName=endpoint_name)
    status = resp["EndpointStatus"]
    logging.info("Status: " + status)

    while status == "Creating":
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
    sm_client.create_endpoint_config(
        EndpointConfigName=endpoint_name,
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

    create_endpoint_response = sm_client.create_endpoint(EndpointName=endpoint_name, EndpointConfigName=endpoint_name)

    print("Endpoint Arn: " + create_endpoint_response["EndpointArn"])


def main(config):
    # Access values from config dictionary
    model_uri = config["model_uri"]
    triton_image_uri = config["triton_image_uri"]
    role = os.getenv("SM_ARN_ROLE") or get_execution_role()  # ''
    env = os.getenv("ENV")
    endpoint_name = config["endpoint_name"]
    endpoint_name = f"{endpoint_name}-{env}"  # add prod/stage to the endpoint for CD in PRs

    sm_client = boto3.client(service_name="sagemaker")

    model_name = get_model_name(model_uri)
    create_model(model_uri, model_name, triton_image_uri, role, sm_client)
    create_endpoint(endpoint_name, model_name, sm_client)
    check_endpoint_initialized(endpoint_name, sm_client)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deploy trained models.")
    parser.add_argument("--config", required=True, help="Path to the configuration YAML file.")

    args = parser.parse_args()

    # Read the YAML file
    with open(args.config, "r") as file:
        config = yaml.safe_load(file)

    main(config)
