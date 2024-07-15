import boto3
import json

import streamlit as st
def main():
    st.set_page_config("QA with Doc")
    st.header("QA with Doc using langchain and AWSBedrock")
    
    user_question=st.text_input("Ask a question from the pdf files")
    
    with st.sidebar:
        st.title("update or create the vector store")
        # if st.button("vectors update"):
        #     with st.spinner("processing..."):
        #         docs=data_ingestion()
        #         get_vector_store(docs)
        #         st.success("done")
        
                
        if st.button("mistral model"):
            with st.spinner("processing..."):
                bedrock_runtime=boto3.client("bedrock-runtime", region_name="ap-south-1")
                # prompt = "where is mumbai located?"

                kwargs = {
                    "modelId": "mistral.mixtral-8x7b-instruct-v0:1",
                    "contentType": "application/json",
                    "accept": "application/json",
                    "body": '{{"prompt":"<s>[INST] {} [/INST]", "max_tokens":200, "temperature":0.5, "top_p":0.9, "top_k":50}}'.format(user_question)
                }

                response=bedrock_runtime.invoke_model(**kwargs)
                body=json.loads(response['body'].read())
                
                print(body['outputs'][0])
                
                st.write(body['outputs'][0])
                st.success("Done")

if __name__=="__main__":
    #this is my main method
    main()