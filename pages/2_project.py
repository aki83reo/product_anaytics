import streamlit as st
import openai 
from langchain.prompts import PromptTemplate
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.chat_models import AzureChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import DocArrayInMemorySearch 
from langchain.chains import RetrievalQA 
from langchain.chains import SequentialChain
from langchain.chains.summarize import load_summarize_chain
import json
#from langchain.llms import AzureOpenAI
#from openai import AzureOpenAI
from langchain.text_splitter import TokenTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from streamlit_chat import message
from langchain.vectorstores import FAISS
import pandas as pd
from langchain.document_loaders import DataFrameLoader
from langchain.document_loaders.csv_loader import CSVLoader
import os
#1.####################################################

def read_config_file(file_path):
    with open(file_path, 'r') as file:
        config = json.load(file)
    return config

 
#2.#####################################################

config_file_path = 'config.json'
config = read_config_file(config_file_path)



openai.api_key = config['api_key']
openai.api_type=config['api_type']
openai.api_base = config['api_base']
openai.api_version = config['api_version']


os.environ['OPENAI_API_KEY']=openai.api_key
os.environ['OPENAI_API_VERSION']=openai.api_version
os.environ['OPENAI_API_BASE']=openai.api_base


import os
os.environ['OPENAI_API_KEY']=openai.api_key

#deployment_name='tnhgpt3'
llm = AzureChatOpenAI(temperature=0.9,deployment_name='gptchat',openai_api_version='2023-05-15', openai_api_base='https://open-ai-demo-tl-001.openai.azure.com/',
openai_api_type='azure')

embedding = OpenAIEmbeddings(deployment="tnhada",openai_api_version='2023-05-15', openai_api_base='https://open-ai-demo-tl-001.openai.azure.com/',
openai_api_type='azure',chunk_size=10)






#file_types=["csv", "pdf", "txt"]
#uploaded_file = st.sidebar.file_uploader("upload", type=file_types, label_visibility="collapsed", accept_multiple_files=True)
#for uploaded_file in uploaded_file:
#uploaded_file.seek(0)
#my_input = pd.read_csv(uploaded_file)
st.session_state.my_input = st.session_state["my_input"]
loader = DataFrameLoader(st.session_state["my_input"], page_content_column="Review")
docs = loader.load()


###########chunking input data 
text_splitter = TokenTextSplitter(chunk_size=10, chunk_overlap=0)
texts = text_splitter.split_documents(docs)
#########################

db = FAISS.from_documents(documents=texts, embedding=embedding)



# Create a conversational chain
chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=db.as_retriever())


# Function for conversational chat
def conversational_chat(query):
    result = chain({"question": query, "chat_history": st.session_state['history']})
    st.session_state['history'].append((query, result["answer"]))
    return result["answer"]

# Initialize chat history
if 'history' not in st.session_state:
    st.session_state['history'] = []

# Initialize messages
if 'generated' not in st.session_state:
    st.session_state['generated'] = ["Hello ! Ask me about " + uploaded_file.name + " 🤗"]

if 'past' not in st.session_state:
    st.session_state['past'] = ["Hey ! 👋"]

# Create containers for chat history and user input
response_container = st.container()
container = st.container()

# User input form
with container:
    with st.form(key='my_form', clear_on_submit=True):
        user_input = st.text_input("Query:", placeholder="Talk to csv data 👉 (:", key='input')
        submit_button = st.form_submit_button(label='Send')

    if submit_button and user_input:
        output = conversational_chat(user_input)
        st.session_state['past'].append(user_input)
        st.session_state['generated'].append(output)

# Display chat history
if st.session_state['generated']:
    with response_container:
        for i in range(len(st.session_state['generated'])):
            message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="big-smile")
            message(st.session_state["generated"][i], key=str(i), avatar_style="thumbs")

