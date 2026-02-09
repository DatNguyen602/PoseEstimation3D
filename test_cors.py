import requests
import json

def test_cors():
    """Test CORS configuration by making a request to the API"""
    api_url = "http://127.0.0.1:8000/"
    
    print("Testing CORS configuration...")
    print(f"Making request to: {api_url}")
    
    # Test with CORS headers
    headers = {
        "Origin": "http://localhost:3000",
        "Access-Control-Request-Method": "POST",
        "Access-Control-Request-Headers": "Content-Type",
    }
    
    try:
        # First try an OPTIONS request (preflight)
        print("\n1. Testing OPTIONS request (preflight)...")
        options_response = requests.options(api_url, headers=headers)
        print(f"OPTIONS Status Code: {options_response.status_code}")
        print(f"CORS Headers: {dict(options_response.headers)}")
        
        # Then try a GET request
        print("\n2. Testing GET request...")
        get_response = requests.get(api_url, headers=headers)
        print(f"GET Status Code: {get_response.status_code}")
        print(f"Response: {get_response.json()}")
        
        # Test with actual POST endpoint
        print("\n3. Testing POST endpoint...")
        test_data = {"message": "Hello from CORS test"}
        post_response = requests.post(
            api_url, 
            json=test_data, 
            headers={"Content-Type": "application/json", "Origin": "http://localhost:3000"}
        )
        print(f"POST Status Code: {post_response.status_code}")
        print(f"Response: {post_response.json()}")
        
        print("\n✅ CORS test completed successfully!")
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Error connecting to API: {e}")
        print("Make sure the API server is running on http://127.0.0.1:8000")

if __name__ == "__main__":
    test_cors()