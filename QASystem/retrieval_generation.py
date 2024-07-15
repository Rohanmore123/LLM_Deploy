# from langchain.chains import RetrievalQA
from langchain.chains.retrieval_qa.base import RetrievalQA
# from langchain.vectorstores import FAISS
# from langchain.llms.bedrock import Bedrock
import boto3
from langchain.prompts import PromptTemplate
# from QASystem.ingestion import get_vector_store
# from QASystem.ingestion import data_ingestion
from ingestion import get_vector_store
from ingestion import data_ingestion
from langchain_community.embeddings import BedrockEmbeddings
from langchain_community.llms.bedrock import Bedrock
from langchain_community.vectorstores import FAISS
from langchain_aws import BedrockLLM

import json

from botocore.exceptions import ClientError

client=boto3.client(service_name="bedrock-runtime",region_name="ap-south-1")
bedrock_embeddings=BedrockEmbeddings(model_id="amazon.titan-embed-text-v1",client=client)
# Set the model ID, e.g., Llama 3 8b Instruct.
model_id = "meta.llama3-8b-instruct-v1:0"

prompt_template = """

Human: Use the following pieces of context to provide a 
concise answer to the question at the end but usse atleast summarize with 
250 words with detailed explaantions. If you don't know the answer, 
just say that you don't know, don't try to make up an answer.
<context>
{context}
</context

Question: {question}

Assistant:"""

PROMPT=PromptTemplate(
    template=prompt_template,input_variables=["context","question"]
)

# Embed the prompt in Llama 3's instruction format.
formatted_prompt = f"""
<|begin_of_text|>
<|start_header_id|>user<|end_header_id|>
{PROMPT}
<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
"""

# Format the request payload using the model's native structure.
native_request = {
    "prompt": formatted_prompt,
    "max_gen_len": 512,
    "temperature": 0.5,
}

# Convert the native request to JSON.
request = json.dumps(native_request)


def get_llama2_llm():
    llm=BedrockLLM(model_id="mistral.mistral-small-2402-v1:0",client=client)
    
    return llm


from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.vectorstores import VectorStoreRetriever

# def get_response_llm(llm,vectorstore_faiss,query):
#     # retriever = VectorStoreRetriever(vectorstore=FAISS(vectorstore_faiss))
#     retriever=vectorstore_faiss.as_retriever(search_type="similarity",search_kwargs={"k":3})
#     retrievalQA = RetrievalQA.from_llm(llm=llm, retriever=retriever)
#     answer=retrievalQA({"query":query})
#     return answer["result"]

def get_response_llm():
    try:
        # Invoke the model with the request.
        response = client.invoke_model(modelId=model_id, body=request)

    except (ClientError, Exception) as e:
        print(f"ERROR: Can't invoke '{model_id}'. Reason: {e}")
        exit(1)

    # Decode the response body.
    model_response = json.loads(response["body"].read())

    # Extract and print the response text.
    response_text = model_response["generation"]
    return response_text

if __name__=='__main__':
    # #docs=data_ingestion()
    # #vectorstore_faiss=get_vector_store(docs)
    # faiss_index=FAISS.load_local("faiss_index",bedrock_embeddings,allow_dangerous_deserialization=True)
    # query="What is RAG token?"
    # llm=get_llama2_llm()
    # print(get_response_llm(llm,faiss_index,query))
    print(get_response_llm)