
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.vectorstores import FAISS
from langchain_community.embeddings import BedrockEmbeddings
# from langchain.llms.bedrock import Bedrock
from langchain_community.llms.bedrock import Bedrock
from langchain_community.vectorstores import FAISS

import json
import os
import sys
import boto3## bedrock client

# bedrock=boto3.client(service_name="bedrock-runtime",region_name='us-east-1')
# bedrock_embeddings=BedrockEmbeddings(model_id="amazon.titan-embed-text-v1",client=bedrock)

bedrock_client = boto3.client(service_name='bedrock-runtime')
bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-text-express-v1",
                                       client=bedrock_client)

def data_ingestion():
    loader=PyPDFDirectoryLoader("./data")
    documents=loader.load()
    
    
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=126, chunk_overlap=126)
    text_splitter.split_documents(documents)
    
    docs=text_splitter.split_documents(documents)
    
    return docs


def get_vector_store(docs):
    vector_store_faiss=FAISS.from_documents(docs,bedrock_embeddings)
    vector_store_faiss.save_local("faiss_index")
    return vector_store_faiss
    
if __name__ == '__main__':
    docs=data_ingestion()
    get_vector_store(docs)
    
    