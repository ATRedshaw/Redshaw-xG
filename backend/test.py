import requests
import json

# Base URL for the Flask API
BASE_URL = "http://127.0.0.1:5000/api/predict"

def test_endpoint(test_case):
    """Send a POST request to the /api/predict endpoint and print the response."""
    print(f"\n=== Test Case: {test_case['description']} ===")
    print("Input:", json.dumps(test_case['input'], indent=2))
    
    try:
        # Send POST request to the endpoint
        response = requests.post(BASE_URL, json=test_case['input'])
        
        # Print response status and content
        print(f"Status Code: {response.status_code}")
        print("Response:", json.dumps(response.json(), indent=2))
        
        # Check if the response matches expected status code
        if response.status_code == test_case['expected_status']:
            print("Test Passed: Status code matches expected.")
        else:
            print(f"Test Failed: Expected status {test_case['expected_status']}, got {response.status_code}")
            
    except requests.exceptions.RequestException as e:
        print(f"Error making request: {str(e)}")

if __name__ == "__main__":
    # Define test cases
    test_cases = [
        # Test 1: Valid basic_model with normalised coordinates
        {
            'description': "Valid basic_model with normalised coordinates",
            'input': {
                'x': 0.5,
                'y': 0.3,
                'situation': None,
                'shot_type': None,
                'normalisation': {'is_normalised': True}
            },
            'expected_status': 200
        },
        # Test 2: Valid basic_model with unnormalised coordinates
        {
            'description': "Valid basic_model with unnormalised coordinates",
            'input': {
                'x': 34.0,
                'y': 52.5,
                'situation': None,
                'shot_type': None,
                'normalisation': {'is_normalised': False, 'max_pitch_width': 68.0, 'max_pitch_length': 105.0}
            },
            'expected_status': 200
        },
        # Test 3: Valid advanced_model with normalised coordinates
        {
            'description': "Valid advanced_model with normalised coordinates",
            'input': {
                'x': 0.8,
                'y': 0.4,
                'situation': 'Penalty',
                'shot_type': 'LeftFoot',
                'normalisation': {'is_normalised': True}
            },
            'expected_status': 200
        },
        # Test 4: Invalid situation
        {
            'description': "Invalid situation",
            'input': {
                'x': 0.5,
                'y': 0.3,
                'situation': 'InvalidSituation',
                'shot_type': None,
                'normalisation': {'is_normalised': True}
            },
            'expected_status': 400
        },
        # Test 5: Invalid shot type
        {
            'description': "Invalid shot type",
            'input': {
                'x': 0.5,
                'y': 0.3,
                'situation': None,
                'shot_type': 'InvalidShotType',
                'normalisation': {'is_normalised': True}
            },
            'expected_status': 400
        },
        # Test 6: Missing x coordinate
        {
            'description': "Missing x coordinate",
            'input': {
                'x': None,
                'y': 0.3,
                'situation': None,
                'shot_type': None,
                'normalisation': {'is_normalised': True}
            },
            'expected_status': 400
        },
        # Test 7: Non-numeric x coordinate
        {
            'description': "Non-numeric x coordinate",
            'input': {
                'x': "invalid",
                'y': 0.3,
                'situation': None,
                'shot_type': None,
                'normalisation': {'is_normalised': True}
            },
            'expected_status': 400
        },
        # Test 8: Invalid normalisation dictionary
        {
            'description': "Invalid normalisation dictionary",
            'input': {
                'x': 0.5,
                'y': 0.3,
                'situation': None,
                'shot_type': None,
                'normalisation': {}
            },
            'expected_status': 400
        },
        # Test 9: Coordinates out of range with is_normalised=True
        {
            'description': "Coordinates out of range with is_normalised=True",
            'input': {
                'x': 70.0,
                'y': 110.0,
                'situation': None,
                'shot_type': None,
                'normalisation': {'is_normalised': True}
            },
            'expected_status': 400
        }
    ]

    # Run each test case
    for test_case in test_cases:
        test_endpoint(test_case)