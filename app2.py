import boto3
import json

bedrock_runtime=boto3.client("bedrock-runtime", region_name="ap-south-1")
prompt = "where is mumbai located?"

kwargs = {
    "modelId": "mistral.mixtral-8x7b-instruct-v0:1",
    "contentType": "application/json",
    "accept": "application/json",
    "body": '{{"prompt":"<s>[INST] {} [/INST]", "max_tokens":200, "temperature":0.5, "top_p":0.9, "top_k":50}}'.format(prompt)
}

response=bedrock_runtime.invoke_model(**kwargs)
body=json.loads(response['body'].read())

print(body['outputs'][0])