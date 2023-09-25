import pytest
from streamlit_app.app import get_endpoint_info

@pytest.mark.parametrize(
    "endpoints, endpoint_name, expected_output",
    [
        ([{"EndpointName":"A1","EndpointURI":"A2"},{"EndpointName":"B1","EndpointURI":"B2"}],
         "A1",
         {"EndpointName":"A1","EndpointURI":"A2"}),
        ([{"EndpointName":"A1","EndpointURI":"A2"},{"EndpointName":"B1","EndpointURI":"B2"}],
         "C1" , None )
    ]
)
def test_get_endpoint_info(endpoints, endpoint_name, expected_output):
   assert expected_output == get_endpoint_info(endpoints, endpoint_name)