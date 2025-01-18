import requests

url = "https://api.getimg.ai/v1/stable-diffusion/text-to-image"


# Replace 'xxx' with your actual API key
api_key = "18yINNTQ4mI3foByU13nhLGV8dM88SpK2oJqruQibVyU9YZvuUlJlPipdJf6pvLUOGklh15fAZ89rpmRclREIEdqu4R7cnTP"

headers = {
    "accept": "application/json",
    "content-type": "application/json",
    "Authorization": f"Bearer key-{api_key}"
}

# The body of the request should include the 'prompt' parameter
data = {
    "prompt": "Create an image of someone named Bilal.",
    "response_format": "url",
    "width": 512,
    "height": 512
}

response = requests.post(url, headers=headers, json=data)

# Print the response text to see the result
print(response.text)
