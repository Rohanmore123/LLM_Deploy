from fastapi import FastAPI, HTTPException, status, Query
from fastapi.responses import JSONResponse, RedirectResponse

import boto3
import json
import os

from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain

from dotenv import load_dotenv
load_dotenv()
from langchain_aws import BedrockLLM
# app = FastAPI()
import os
import boto3
import json
from fastapi import HTTPException


load_dotenv()

# Setup AWS credentials and Bedrock client
ROLE_ARN = os.getenv('ROLE_ARN')
AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')


lambda_client = boto3.client(
    'lambda',
    region_name=os.getenv('REGION_NAME'),
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY')
)

def get_context(question: str):
    try:
        # Invoke the Lambda function
        response = lambda_client.invoke(
            FunctionName='rag-pdf',
            InvocationType='RequestResponse',  # Changed to RequestResponse
            Payload=json.dumps({"Question": question})
        )

        # print(response)
        
        # Read the Lambda function's response stream and parse it
        response_payload = response['Payload'].read()
        # print(response_payload)
        # Check if the response payload is not empty
        if not response_payload:
            raise ValueError("Empty response from Lambda function")
        
        response_payload_dict = json.loads(response_payload)
        # print(response_payload_dict)
        
        # Navigate to the retrievalResults
        results = response_payload_dict['body']['answer']['retrievalResults']

        # Initialize an empty string to store the extracted paragraph
        extracted_paragraph = ""
        
        # Loop through each result and concatenate text to a paragraph
        for result in results:
            text = result['content']['text']
            extracted_paragraph += text + " "
        # print(extracted_paragraph)
        # Return the concatenated paragraph
        return {"response": extracted_paragraph.strip()}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def get_answer_from_kb(query: str):
    bedrock_client = boto3.client(
            "bedrock-runtime",
            region_name="us-east-1",
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY)
    # bedrock=boto3.client(service_name="bedrock-runtime",region_name="us-east-1")
    llm=BedrockLLM(model_id="mistral.mixtral-8x7b-instruct-v0:1",client=bedrock_client,model_kwargs={'max_tokens_to_sample': 512})
    
    kb_prompt_template = """
    You are a helpful AI assistant who is expert in answering questions. Your task is to answer user's questions as factually as possible. You will be given enough context with information to answer the user's questions. Find the context:
    Context: {context}
    Question: {query}

    Now generate a detailed answer that will be helpful for the user. Return the helpful answer.

    Answer: 
    """

    prompt_template_kb = PromptTemplate(
        input_variables=["context", "query"], template=kb_prompt_template
    )

    llm_chain = LLMChain(llm=llm, prompt=prompt_template_kb)
    
    context = get_context(query)
    print(context)
    result = llm_chain.invoke({"context":context, "query":query})

    return result

query= "what is the fifth step towards riches?"

response = get_answer_from_kb(query)
print(JSONResponse(content=response, status_code=200))
