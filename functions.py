########################## Packages ########################
import openai
import os
import pandas as pd
import json
from flask import request,jsonify,render_template
from langchain.chat_models import ChatOpenAI
from langchain.chat_models import AzureChatOpenAI
from langchain.embeddings import OpenAIEmbeddings

from langchain.prompts import PromptTemplate
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.chat_models import AzureChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.chains import RetrievalQA
from langchain.chains import SequentialChain
from langchain.chains.summarize import load_summarize_chain
import streamlit as st
import os
from PyPDF2 import PdfReader
import docx
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI

from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from streamlit_chat import message
from langchain.callbacks import get_openai_callback
###############################################################
#1.####################################################

def read_config_file(file_path):
    with open(file_path, 'r') as file:
        config = json.load(file)
    return config

 
#2.#####################################################

config_file_path = 'config.json'
config = read_config_file(config_file_path)

#3.########################################################

 

openai.api_key = config['api_key']
openai.api_type=config['api_type']
openai.api_base = config['api_base']
openai.api_version = config['api_version']

import os
os.environ['OPENAI_API_KEY']=openai.api_key
#4.########################################################

llm = AzureChatOpenAI(temperature=0.9,deployment_name='',openai_api_version='', openai_api_base='',
openai_api_type='')

embedding = OpenAIEmbeddings(deployment="",openai_api_version='', openai_api_base='',
openai_api_type='',chunk_size=10)

########################################################
#4. ##################summary of reviews#####################################

 

def review_summary(document):

 
    query="""

        You are a product analyst, follw the below steps:
         1. Analyze all reviews.
         2. Extract and summarize the important aspects.
         3. Put the summary in a list format.
         4. Remove extra text, dont add extra text.
         5.The resultshould be like ["the toothpaste is good", "all is good"]
    

    """

 
    chain = load_summarize_chain(llm, chain_type="stuff")
    response = chain.run(document).split('.')

    return response

#########################################################################

def aspect_extraction(documents):
    all_reviews = "".join([documents[i].page_content for i in range(len(documents))])
    aspect_extraction_prompt1 = """ Analyse all the reviews in the text below and follow the below steps to give the output.
                            Step1 : Extract all the key aspects of the product or service that is expressed in the reviews.
                            Step2 : Group all the aspects that pertains to a common attribute of the product or service into a single set.
                            Step3 : For each set of aspects identified in step 2 assign a attribute name that best captures the feature of the product described by the aspects.
                            Step4 : Return the list of attribute names from step 3, aspects should be unique and max 10,avoid extra texts.
                            Step5 : Dont add extra text, it should return only a list.
                            text : {all_reviews}

                            """

    aspect_extraction_prompt2 = """ Analyse the list of aspects and perform the below actions :
                            Step1 : Group the aspects that are related to the same benefit or objective offered by the product into a single set.
                            Step2 : For each set obtained in step 1 assign a attribute name that captures the benefit theme
                            Step3 : Return the list of attribute names from step 2
                            format the output as a python list
                            text : {stage1_aspects}

                            """

 

    first_prompt = ChatPromptTemplate.from_template(aspect_extraction_prompt1)
    chain_one = LLMChain(llm=llm, prompt=first_prompt, output_key="stage1_aspects")

 

    second_prompt = ChatPromptTemplate.from_template(aspect_extraction_prompt2)
    chain_two = LLMChain(llm=llm, prompt=second_prompt,output_key="aspects_final")

 
    overall_chain = SequentialChain(
        chains=[chain_one, chain_two],
        input_variables=["all_reviews"],
        output_variables=["stage1_aspects", "aspects_final"],
        verbose=False

    )

 

    aspects_output = overall_chain(all_reviews)["aspects_final"]

 

    if type(aspects_output)==str:

        final_list_aspects=eval(aspects_output)

    else:

        final_list_aspects=aspects_output



    return final_list_aspects



#6. ############### Sentiments wrt aspects for all reviews ####################

def sentiment_aspects(documents,document_number,aspects_output):

 

    query1=f"""map text in a : {aspects_output} with individual  {documents[document_number]}, and follow below steps:

            1. aspect wise sentiments postive,negative or neutral for the doc in a list format.

            2. store the result in a dictionery  no extra text, should be in a dictionery,with keys as the aspects, not a string.

            3.just provide the list, example ['positive','negative',....]

            4. No extra texts

            5. Just return the dictionery, please maintain the dictionery format.

"""

   

    response = eval(llm.call_as_llm(query1))
    response_df=pd.DataFrame.from_dict([response])
    filename_review=list(documents[document_number].page_content.split("\n"))
    response_df['Filename']=filename_review[0]
    response_df['Review']=filename_review[1]

 

    return response_df

#7. ################# count of sentiments by aspects ###########################

def count_sentiments(column):
        positive_count=(column=='positive').sum()
        negative_count=(column=='negative').sum()
        neutral_count=(column=='neutral').sum()
        return { 'Positive':positive_count,'Negative':negative_count,'Neutral':neutral_count}


def count_sentis(df):
    sentiment_counts={}
    df.drop(columns=['Filename','Review'],inplace=True)
    for column in df.columns:
        sentiment_counts[column]=count_sentiments(df[column])
    sentiment_count_df=pd.DataFrame(sentiment_counts).T
    return sentiment_count_df
 
 
#8. ################# Load the llm model  ###########################

