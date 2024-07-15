import os
import json
import boto3
from dotenv import load_dotenv
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import BedrockEmbeddings
from langchain_community.llms.bedrock import Bedrock
from langchain_community.vectorstores.faiss import FAISS
from langchain_community.vectorstores import faiss
import os
from typing import Optional

# External Dependencies:
import boto3
from botocore.config import Config
# Load environment variables
load_dotenv()

# Setup AWS credentials and Bedrock client
ROLE_ARN = os.getenv('ROLE_ARN')
# AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
# AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
# print(AWS_ACCESS_KEY_ID)
sts_client = boto3.client("sts")
assumed_role = sts_client.assume_role(RoleArn=ROLE_ARN, RoleSessionName="bedrock-runtime")
credentials = assumed_role["Credentials"]
# print(credentials)
session_kwargs = {"region_name": 'ap-south-1'}
client_kwargs = {**session_kwargs}
session_kwargs["profile_name"] = 'default'
retry_config = Config(
        region_name='ap-south-1',
        retries={
            "max_attempts": 10,
            "mode": "standard",
        },
    )
session = boto3.Session(**session_kwargs)
sts = session.client("sts")
response = sts.assume_role(
                RoleArn="arn:aws:iam::471112983796:role/EC2_full_Access",
                RoleSessionName="bedrock-runtime",
            )
def _get_bedrock_client( role_arn):
        

        bedrock_client = boto3.client(
            "bedrock-runtime",
            region_name="ap-south-1",
            aws_access_key_id=credentials["AccessKeyId"],
            aws_secret_access_key=credentials["SecretAccessKey"],
            aws_session_token=credentials["SessionToken"]
        )
        return bedrock_client
bedrock_client = session.client(service_name="bedrock-runtime",config=retry_config, **client_kwargs )

# bedrock_runtime=boto3.client("bedrock-runtime",
#                               region_name="ap-south-1",
#                              aws_access_key_id=credentials["AccessKeyId"],
#                              aws_secret_access_key=credentials["SecretAccessKey"],
#                               aws_session_token=credentials["SessionToken"])

bedrock_runtime=boto3.client(service_name="bedrock-runtime", region_name='ap-south-1')
loader = PyPDFLoader("D:/Projects/RAG_ops/chains/attention.pdf")
documents = loader.load()
# print(f"Loaded {len(documents)} documents")

# # Print out the type and contents of a few documents to debug
# for i, doc in enumerate(documents[:3]):
#     print(f"Document {i}: Type: {type(doc)}, Content: {doc.page_content[:100]}...")

# Split documents into smaller chunks
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50, separator="\n")
docs = text_splitter.split_documents(documents=documents)
# print(f"Split into {len(docs)} chunks")

# # Extract text from documents for embedding
# texts = [doc.page_content for doc in docs]
# print(type(texts))


# texts = [doc.page_content for doc in docs]


# def embed(text):
#     kwargs = {
#     "modelId": "amazon.titan-embed-text-v1",
#     "contentType": "*/*",
#     "accept": "*/*",
#     # "body": '{{"prompt":"<s>[INST] {} [/INST]", "max_tokens":200, "temperature":0.5, "top_p":0.9, "top_k":50}}'.format(docs)
#     "body": {"input_text":doc}
#     }
#     resp=bedrock_runtime.invoke_model(**kwargs)
#     resp_body=json.loads(resp.get('body').read())
#     return resp_body.get('embedding')
# import numpy as np
# embedding_array=np.array([]).reshape(0,3000)
 
# for doc in docs:
#     embeddings=embed(doc)
#     embedding_array=np.append(embedding_array, np.array(embeddings).reshape(1,-1),axis=0)

# index=faiss.IndexFlatL2(3000)
# index.add(embedding_array)
# print(index.ntotal)
# # kwargs = {
# #     "modelId": "amazon.titan-embed-text-v1",
# #     "contentType": "application/json",
# #     "accept": "application/json",
# #     # "body": '{{"prompt":"<s>[INST] {} [/INST]", "max_tokens":200, "temperature":0.5, "top_p":0.9, "top_k":50}}'.format(docs)
# #     "body": {"input_text":docs}
# # }
# # # bedrock_runtime=bedrock_runtime.invoke_model(**kwargs)

# # # print(bedrock_runtime)
# # bedrock_embeddings=BedrockEmbeddings(model_id="amazon.titan-embed-text-v1",client=bedrock_runtime,model_kwargs=kwargs)
# # vectorstore_faiss = FAISS.from_documents(docs, bedrock_embeddings,**kwargs,allow_dangerous_deserialization=True)
# # vectorstore_faiss.save_local("D:\Projects\RAG_ops\\faissss")

# # # faiss_index=FAISS.load_local("faissss",bedrock_embeddings,allow_dangerous_deserialization=True)

# # # vectorstore_faiss = FAISS.from_documents(docs, bedrock_embeddings)
# # # vectorstore_faiss.save_local("index_store")
# # # print(bedrock_embeddings)

import json
import boto3

class TitanEmbeddings(object):
    accept = "application/json"
    content_type = "application/json"
    
    def __init__(self, model_id="amazon.titan-embed-text-v1", boto3_client=bedrock_runtime, region_name='ap-south-1'):
        
        self.bedrock_boto3 = boto3_client
        
        self.model_id = model_id

    def __call__(self, text, dimensions, normalize=True):
        """
        Returns Titan Embeddings

        Args:
            text (str): text to embed
            dimensions (int): Number of output dimensions.
            normalize (bool): Whether to return the normalized embedding or not.

        Return:
            List[float]: Embedding
            
        """

        body = json.dumps({
            "inputText": text,
            "dimensions": dimensions,
            "normalize": normalize
        })

        response = self.bedrock_boto3.invoke_model(
            body=body, modelId=self.model_id, accept=self.accept, contentType=self.content_type
        )

        response_body = json.loads(response.get('body').read())

        return response_body['embedding']
    
import json
import os
import sys

import boto3

# boto3_bedrock_runtime = get_bedrock_client() #boto3.client('bedrock')

bedrock_embeddings = TitanEmbeddings(model_id="amazon.titan-embed-text-v1", boto3_client=_get_bedrock_client(role_arn="arn:aws:iam::471112983796:role/EC2_full_Access"))
print(bedrock_embeddings)
from langchain.prompts import PromptTemplate
prompt_template = """

Human: Use the following pieces of context to provide a 
concise answer to the question at the end but usse atleast summarize with 
250 words with detailed explainations. If you don't know the answer, 
just say that you don't know, don't try to make up an answer.
<context>
{context}
</context>

Question: {question}

Assistant:"""

PROMPT=PromptTemplate(
    template=prompt_template,input_variables=["context","question"]
)



modelId = "meta.llama3-8b-instruct-v1:0"  # 
accept = "*"
contentType = "*"
prompt_data = "Amazon Bedrock supports foundation models from industry-leading providers such as \
AI21 Labs, Anthropic, Stability AI, and Amazon. Choose the model that is best suited to achieving \
your unique goals."

kwargs = {
        "modelId": "mistral.mixtral-8x7b-instruct-v0:1",
        "contentType": "application/json",
        "accept": "application/json",
        "body": json.dumps({
            "prompt": "where is mumbai located?",
            "max_tokens": 200,
            "temperature": 0.5,
            "top_p": 0.9,
            "top_k": 50
        })
    }
sample_model_input={
    "inputText": PROMPT,
    "dimensions": 256,
    "normalize": True
}
body= ({
        "prompt": PROMPT,
        "max_tokens": 200,
        "temperature": 0.5,
        "top_p": 0.9,
        "top_k": 50
        })
# body = json.dumps(sample_model_input)
boto3_bedrock_runtime=_get_bedrock_client(role_arn="arn:aws:iam::471112983796:role/EC2_full_Access")

# response = boto3_bedrock_runtime.invoke_model(body=body, modelId=modelId, accept=accept, contentType=contentType)
response = boto3_bedrock_runtime.invoke_model(**kwargs)

response_body = json.loads(response.get('body').read())
# print(response_body)
# embedding = response_body.get("embedding")
# print(f"The embedding vector has {len(embedding)} values\n{embedding[0:3]+['...']+embedding[-3:]}")
bedrock_client = boto3.client(service_name='bedrock-runtime')
bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-text-express-v1",
                                       client=bedrock_client)

# from langchain_community.document_loaders import PyPDFDirectoryLoader
# loader=PyPDFDirectoryLoader("./data")
# documents=loader.load()
# from langchain.text_splitter import RecursiveCharacterTextSplitter

# text_splitter=RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=250)
# text_splitter.split_documents(documents)

# docs=text_splitter.split_documents(documents)

# vector_store_faiss=FAISS.from_documents(docs,bedrock_embeddings)
# vector_store_faiss.save_local("faiss_index","faissss")
from langchain_aws import BedrockLLM

llm=BedrockLLM(model_id="mistral.mixtral-8x7b-instruct-v0:1",client=boto3_bedrock_runtime,model_kwargs={'max_tokens_to_sample': 512})

from langchain.chains.retrieval_qa.base import RetrievalQA
faiss_index=FAISS.load_local("faiss_index",bedrock_embeddings,allow_dangerous_deserialization=True)
print(faiss_index)


prompt_template = """

    Human: Please use the given context to provide concise answer to the question
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    <context>
    {context}
    </context>

    Question: {question}

    Assistant:"""

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)




from langchain_core.prompts import ChatPromptTemplate
system_prompt = (
    "Use the given context to answer the question. "
    "If you don't know the answer, say you don't know. "
    "Use three sentence maximum and keep the answer concise. "
    "Context: {context}"
)
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

retriever=faiss_index.as_retriever()
question_answer_chain = create_stuff_documents_chain(llm, prompt)
# print(question_answer_chain)
chain = create_retrieval_chain(retriever, question_answer_chain)
# print(chain)
query="What is RAG token?"
# cont="explain RAG related"
# answer=chain.invoke({"input": query})[0]


qa = RetrievalQA.from_chain_type(
llm=llm,
chain_type="stuff",
retriever=faiss_index.as_retriever(
    search_type="similarity", search_kwargs={"k": 5}
),
return_source_documents=True,
chain_type_kwargs={"prompt": prompt}
)
# print(qa)
query="What is success"

result_ = qa.invoke({"query":query})

result = result_["result"].strip()
print(result)
# cont="about success"
# answer=qa({"query":query})
# print(answer["result"])
# print(retriever.invoke(query))
# docu=faiss_index.similarity_search(query)
# print(docu[0].page_content)