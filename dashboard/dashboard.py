
import streamlit as st
import pandas as pd

st.title("Hello")
st.markdown(
    """ 
    This is a dashboard for Data-Science-Switching-Behaviour-in-Statutory-Health-Insurance. 

**Project Motivation**

- Health insurers face increasing churn in competitive markets.

- ML is used to predict customer switching, but explanations are often too technical for stakeholders.

- There’s a need for clear, human-understandable insights—especially in sensitive domains like healthcare.

- We explore how Large Language Models (LLMs) can bridge this gap by translating complex XAI outputs into accessible narratives.
    """
)

# File uploader widget
uploaded_file = st.file_uploader("Choose a file", type=["xlsx"])

if uploaded_file is not None:
    # To read file as bytes:
    bytes_data = uploaded_file.getvalue()
    st.write(bytes_data)

    # To convert to a string based IO:
    stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
    st.write(stringio)

    # To read file as string:
    string_data = stringio.read()
    st.write(string_data)

    # To read file as dataframe:
    if uploaded_file.name.endswith('.xlsx'):
        # df = pd.read_csv(uploaded_file)
        df = pd.read_excel(uploaded_file, sheet_name=0, header=None)
        st.write(df)
        # todo process file
    else:
        st.write("Please upload an excel file to display as a dataframe.")