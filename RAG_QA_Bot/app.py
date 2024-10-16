import databutton as db
import streamlit as st
import cohere
from functions import get_index_for_pdf
from langchain.chains import RetrivalAQ
from langchain.llms import cohere
import os
from dotenv import load_dotenv

load_dotenv()

st.title("RAG QA ChatBot")

os.environ['Cohere_api'] = db.secrets.get("Cohere_api")
cohere.api_key = db.secrets.get("Cohere_api")


@st.cache_data

def create_vectordb(files,filenames):
    with st.spinner("Vector database"):
            vectordb = get_index_for_pdf(
                  [file.getvalue() for file in files], filenames, cohere.api_key
                  
            )