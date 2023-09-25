def generate_triton_config_mnist(model_path, model_name):
    """Generate config.pbtxt for Triton Inference Server."""

    config_content = f"""
    name: "{model_name}"
    platform: "tensorflow_savedmodel"
    max_batch_size: 0

    input {{
    name: "input_1"
    data_type: TYPE_FP32
    dims: [ -1, 28, 28, 1 ]
    }}

    output {{
    name: "output_1"
    data_type: TYPE_FP32
    dims: [ -1, 10 ]
    }}
        """

    config_path = f"{model_path}/config.pbtxt"

    with open(config_path, "w") as config_file:
        config_file.write(config_content)

    print(f"Config written to: {config_path}")
