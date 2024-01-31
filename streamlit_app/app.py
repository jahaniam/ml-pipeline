import os
import numpy as np
import cv2
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import boto3
import json


def get_endpoint_info(endpoints, endpoint_name):
    try:
        for endpoint in endpoints:
            if endpoint["EndpointName"] == endpoint_name:
                return endpoint  # return the matching endpoint dictionary
    except:
        return None


def predict(img, runtime_sm_client, endpoint_name):
    payload = {
        "inputs": [
            {
                "name": "input_1",
                "shape": [1, 28, 28, 1],
                "datatype": "FP32",
                "data": img.tolist(),
            }
        ]
    }
    response = runtime_sm_client.invoke_endpoint(
        EndpointName=endpoint_name, ContentType="application/octet-stream", Body=json.dumps(payload)
    )
    result = json.loads(response["Body"].read().decode("utf8"))
    probabilities = np.array(result["outputs"][0]["data"]).reshape(result["outputs"][0]["shape"])[0]
    label = np.argmax(probabilities)
    return label


@st.cache_resource
def get_session():
    runtime_sm_client = boto3.client("sagemaker-runtime")
    return runtime_sm_client


@st.cache_data
def get_endpoint_names():
    sm_client = boto3.client("sagemaker")
    response = sm_client.list_endpoints(
        SortBy="CreationTime",
        SortOrder="Descending",
    )
    endpoints = response["Endpoints"]
    return endpoints


def main():
    session = get_session()

    st.title("😎 Digit Classification")
    st.write("Draw a digit and click predict")

    col1, col2 = st.columns(2)

    SIZE = 256
    with col1:
        canvas_result = st_canvas(
            fill_color="#000000",
            stroke_width=20,
            stroke_color="#FFFFFF",
            background_color="#000000",
            width=SIZE,
            height=SIZE,
            drawing_mode="freedraw",
            key="canvas",
        )

    endpoints = get_endpoint_names()
    endpoint_names = [endpoint["EndpointName"] for endpoint in endpoints]
    selected_endpoint_name = st.sidebar.selectbox("Endpoint", endpoint_names)

    with col2:
        if st.button("Predict"):
            if selected_endpoint_name is None:
                st.error(f"Please select an endpoint")
            else:
                img = cv2.resize(canvas_result.image_data.astype("uint8"), (28, 28))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                label = predict(img.reshape(1, 28, 28, 1), session, endpoint_name=selected_endpoint_name)
                st.text(f"Prediction: {label}")

    with st.sidebar, st.expander("Status", expanded=True):
        selected_endpoint = get_endpoint_info(endpoints, selected_endpoint_name)
        if selected_endpoint is None:
            st.error(f"Endpoint not found")
        else:
            st.json(selected_endpoint)


if __name__ == "__main__":
    main()
