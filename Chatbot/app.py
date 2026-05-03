# from langchain_openai import ChatOpenAI 
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama 
import streamlit as st 
from dotenv import load_dotenv
import os

load_dotenv()

## prompt template

prompt = ChatPromptTemplate.from_messages(
    [
        ("system","You are helpful assistant. Please response to the user queries"),
        ("user", "Question:{question}")
    ]
)

## streamlit framework

st.title('Langchain Demo with Ollama')
input_txt = st.text_input("Search the topic you want !!!")

# Open source LLM
llm = ChatOllama(model="llama3:latest")

output_parser = StrOutputParser()
chain = prompt| llm | output_parser

if input_txt:
    st.write(chain.invoke({'question':input_txt}))

