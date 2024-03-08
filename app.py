from operator import truediv
import streamlit as st
import pandas as pd
import pdfplumber
import os
import numpy as np
import plotly.express as px
from functions import review_summary,aspect_extraction,sentiment_aspects,count_sentis
from langchain.document_loaders import CSVLoader
from langchain.document_loaders import DataFrameLoader
 
 
st.set_page_config(
    page_title="Product Dashboard",
    page_icon=":bar_chart:",
    layout="wide"
)
 
  
# ------MAINPAGE-----------
st.markdown("""
        <h2 style='text-align: center;'> Product Dashbaord 📊</h1>
        """,
    unsafe_allow_html=True,
        )
st.markdown("---")
 
 
# -------TOP KPIs ------------------
 
total_reviews=100
total_positive=45
total_negative=32
total_neutral=33
 
left_column,middle_column,right_column, final_right=st.columns(4)
with left_column:
    st.subheader("Total Reviews : ")
    st.subheader(f"{total_reviews:,}")
 
with middle_column:
    st.subheader("Total Positive Reviews :")
    st.subheader(f"{total_positive:,}")
 
with right_column:
    st.subheader("Total Negative Reviews :")
    st.subheader(f"{total_neutral:,}")
 
with final_right:
    st.subheader("Total Neutral Reviews :")
    st.subheader(f"{total_neutral:,}")
st.markdown("---")


def init_session_state():
    if "docs" not in st.session_state:
        st.session_state.docs = ""
    if "my_input" not in st.session_state:
        st.session_state.my_input = ""
    if "summary" not in st.session_state:
        st.session_state.summary = ""

# Initialize session state
init_session_state()
 

file_types=["csv", "pdf", "txt"]
uploaded_file = st.sidebar.file_uploader("upload", type=file_types, label_visibility="collapsed", accept_multiple_files=True)
for uploaded_file in uploaded_file:
    uploaded_file.seek(0)
    if "my_input" not in st.session_state:
       st.session_state["my_input"]=""
    my_input = pd.read_csv(uploaded_file)
    st.session_state.my_input = my_input
    loader = DataFrameLoader(my_input, page_content_column="Review")
    docs = loader.load()
    st.session_state.docs = docs


 
css = '''
<style>
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
    font-size:1.5rem;
    }
</style>
'''
 
st.markdown(css, unsafe_allow_html=True)
st.markdown(
    """
    <style>
        button[data-baseweb="tab"] {
        font-size: 60px;
        margin: 0;
        font-weight: 400px;
        width: 100%
        }
    </style>
    
    """,
    unsafe_allow_html=True,
    )

def sentiment_count(docs,aspects):

    sentiment_list=[]
    count=0
    concat_sentiment_reviews=0
    for i in range(len(docs)):
            count = count + 1
            print(count)
            try:
                sentiment_list.append(sentiment_aspects(docs, i, aspects))
            except:
                pass
                
    concat_sentiment_reviews = pd.concat(sentiment_list)
    count_of_sentiments_vals = count_sentis(concat_sentiment_reviews)
    sentiment_counts = count_of_sentiments_vals.reset_index()
    melted_df = pd.melt(sentiment_counts, id_vars='index', var_name='Feature', value_name='Count')
    print(melted_df,"melted_df")
    return count_of_sentiments_vals,melted_df





tab1, tab2,tab3= st.tabs(["✍ Get Summary and aspects","📈 Chart","🤖 Chat with your data"])
import time

with tab1:
     
        col1,col2=st.columns(2)
    
        with col1:
            with st.container():
                st.header("🗃 Summary")
                with st.spinner('Loading summary...'): 
                    summary = review_summary(st.session_state.docs),
                    st.session_state.summary = summary[0]
                    filtered_summary = [item for item in summary[0] if item.strip()]
                    numbered_bullet_points_summary = "\n".join([f"{i+1}. {item}<br>" for i, item in enumerate(filtered_summary)])
                    st.markdown(
                            f"<div style='font-size:22px; font-family: Arial, sans-serif; margin-top: 40px;'>{numbered_bullet_points_summary}</div>",
                        unsafe_allow_html=True)
                    
                time.sleep(5)
                st.success('Done!')

                          
        with col2:
              
            with st.container():
                st.header("🗃 Important Aspects")
                with st.spinner('Loading Aspects...'): 
                    summary_aspects = aspect_extraction(st.session_state.docs),
                    filtered_summary = [item for item in summary_aspects[0] if item.strip()]
                    numbered_bullet_points_summary = "\n".join([f"{i+1}. {item}<br>" for i, item in enumerate(filtered_summary)])
                    st.markdown(
                            f"<div style='font-size:22px; font-family: Arial, sans-serif; margin-top: 40px;'>{numbered_bullet_points_summary}</div>",
                        unsafe_allow_html=True)
                time.sleep(5)
                st.success('Done!')                


result_barchart = sentiment_count(docs,summary_aspects)[0]
fig = px.bar(result_barchart, x=result_barchart.index, y=result_barchart.columns,
                     barmode='stack', labels={'index': 'Sentiments', 'value': 'distribution'},
                     title='Sentiments by aspects')
fig.update_layout(
    height=900,  # set the height in pixels
    width=900    # set the width in pixels
)


result_bubblechart = sentiment_count(docs,summary_aspects)[1]

#melted_df = pd.melt(result_bubblechart, id_vars='index', var_name='Feature', value_name='Count')

# Create a bubble plot with sentiments on the axis
fig_bubble = px.scatter(result_bubblechart, x='Feature', y='index', size='Count', color='Feature',
                        labels={'Feature': 'Sentiment', 'index': 'Aspect', 'Count': 'Number of Reviews'},
                        title='Sentiments for Each Aspect (Bubble Plot)')


fig_bubble.update_layout(height=900, width=900)

with tab2:
   col1,col2=st.columns(2)

   col1.plotly_chart(fig)
   col2.plotly_chart(fig_bubble)





    