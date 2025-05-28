import os

from dotenv import load_dotenv
load_dotenv()

from langchain_community.llms import Ollama

import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Langsmith tracking

os.environ['LANGCHAIN_API_KEY']=os.getenv('LANGCHAIN_API_KEY')
os.environ['LANGCHAIN_PROJECT']=os.getenv('LANGCHAIN_PROJECT')

os.environ['LANGCHAIN_TRACING_V2']='True'

# Prompt Template

prompt=ChatPromptTemplate(
    [
        ('system', "you are a helpful assistant. Please respond to the question asked"),
        ('user', 'Question:{question}')
    ]
)

# Streamlit Framework

st.title("Langchain Demo with GEMMA")
input_text=st.text_input("What question you have in your mind?")

# OLLAMA GEMMA Model

llm=Ollama(model='gemma:2b')
output_parser=StrOutputParser()
chain=prompt|llm|output_parser

if input_text:
    st.write(chain.invoke({"question":input_text}))

