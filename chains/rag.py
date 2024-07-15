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

# Load environment variables
load_dotenv()

# Setup AWS credentials and Bedrock client
ROLE_ARN = os.getenv('ROLE_ARN')
AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')

class VectorSearchWithBedrock():
    def __init__(self, index_store="./index/faiss_index"):
        self.module = "__name__"
        self.bedrock_client = self._get_bedrock_client(ROLE_ARN)
        self.embeddings = BedrockEmbeddings(client=self.bedrock_client, region_name="ap-south-1")
        self.index_store = index_store

    def _get_bedrock_client(self, role_arn):
        sts_client = boto3.client("sts")
        assumed_role = sts_client.assume_role(RoleArn=role_arn, RoleSessionName="bedrock-session")
        credentials = assumed_role["Credentials"]

        bedrock_client = boto3.client(
            "bedrock-runtime",
            region_name="ap-south-1",
            aws_access_key_id=credentials["AccessKeyId"],
            aws_secret_access_key=credentials["SecretAccessKey"],
            aws_session_token=credentials["SessionToken"]
        )
        return bedrock_client

    def save_into_vector(self, docs):
        vectorstore_faiss = FAISS.from_documents(docs, self.embeddings)
        vectorstore_faiss.save_local(self.index_store)

    def load_vector(self,index):
        faiss_vectorstore = FAISS.load_local(index, self.embeddings,allow_dangerous_deserialization=True)
        return faiss_vectorstore

    def return_bedrock_llm(self, model_id="mistral.mixtral-8x7b-instruct-v0:1"):
        llm = Bedrock(client=self.bedrock_client, model_id=model_id)
        return llm

if __name__ == "__main__":
    # Load and split PDF document
    loader = PyPDFLoader("D:/Projects/RAG_ops/chains/attention.pdf")
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50, separator="\n")
    docs = text_splitter.split_documents(documents=documents)

    # Save vectors to local FAISS store
    faiss_chat_bot = VectorSearchWithBedrock()
    # faiss_chat_bot.save_into_vector(docs)

    # Load vector store and query for relevant documents
    wrapper_faiss_chat = VectorStoreIndexWrapper(vectorstore=faiss_chat_bot.load_vector("D:\Projects\RAG_ops\\faiss_index"))
    llm=faiss_chat_bot.return_bedrock_llm()
    # Use Bedrock model to generate response
    query = "where is mumbai located?"
    relevant_docs = wrapper_faiss_chat.query(query,llm)
    context = " ".join([doc.page_content for doc in relevant_docs])

    prompt = f"<s>[INST] {context}\n\n{query} [/INST]"

    kwargs = {
        "modelId": "mistral.mixtral-8x7b-instruct-v0:1",
        "contentType": "application/json",
        "accept": "application/json",
        "body": json.dumps({
            "prompt": prompt,
            "max_tokens": 200,
            "temperature": 0.5,
            "top_p": 0.9,
            "top_k": 50
        })
    }

    response = faiss_chat_bot.bedrock_client.invoke_model(**kwargs)
    body = json.loads(response['body'].read())

    # Remove 'stop_reason' key from the response
    output = body['outputs'][0]
    output.pop('stop_reason', None)

    print(output['text'])
