import pytest
from sagemaker_scripts.deploy_sagemaker import check_model_name, get_model_name

@pytest.mark.parametrize(
    "model_uri, expected_model_name", 
    [
        (
            's3://sagemaker-us-east-1-633875729936/digit_classification/models/tensorflow-training-2023-09-25-05-55-53-638/model.tar.gz', 
            'tensorflow-training-2023-09-25-05-55-53-638'
        ),
        (
            's3://sagemaker-us-east-1-633875729936/digit_classification/models/tensorflow-training-A/output/model.tar.gz', 
            'tensorflow-training-A'
        ),
    ]
)
def test_get_model_name(model_uri, expected_model_name):
    """
    Validates if get model works given different type of model_uri
    """
    model_name = get_model_name(model_uri)
    assert model_name == expected_model_name