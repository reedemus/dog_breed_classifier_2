import boto3
import json
import base64
 
image = 'australian-terrier.jpg'
endpoint = 'insert name of your endpoint here'
 
runtime = boto3.Session().client('sagemaker-runtime')
 
# Read image into memory
with open(image, 'rb') as f:
    payload = base64.b64encode(f.read())
# Send image via InvokeEndpoint API
response = runtime.invoke_endpoint(EndpointName=endpoint, ContentType='application/x-image', Body=payload)

# Unpack response
result = json.loads(response['Body'].read().decode())
