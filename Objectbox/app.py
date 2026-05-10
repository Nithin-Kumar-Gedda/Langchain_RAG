import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_objectbox.vectorstores import ObjectBox
from langchain_community.document_loaders import PyPDFDirectoryLoader

from dotenv import load_dotenv
load_dotenv()
grop_api_key = os.environ['GROQ_API_KEY']

llm = ChatGroq(groq_api_key=grop_api_key,
               model="llama-3.1-8b-instant")

st.title("Objectbox DB with Llama3 Demo")

prompt = ChatPromptTemplate.from_template(
"""
Answer the questions based on the provided context only.
Please provide the most accurate response based on the question
<context>
{context}
</context>
Question : {input}
"""
)

## vector embedding

def vec_embedding():
    if "vectors" not in st.session_state:
        st.session_state.embeddings = OllamaEmbeddings(model="llama3:latest")
        st.session_state.loader = PyPDFDirectoryLoader("../Llama Agent/data") # data ingestion
        st.session_state.docs = st.session_state.loader.load() #docs loading
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.final_docs = st.session_state.text_splitter.split_documents(st.session_state.docs[:30])
        st.session_state.vectors = ObjectBox.from_documents(st.session_state.final_docs, st.session_state.embeddings,embedding_dimensions=768)


input_prompt = st.text_input("Enter your Question from Documents")

if st.button("Document Embedding"):
    vec_embedding()
    st.write("Vector score database is ready!!!")

if input_prompt:
    doc_chain = prompt| llm | StrOutputParser()
    retriever = st.session_state.vectors.as_retriever()
    retrievel_chain = (
        {"context":retriever, "input": RunnablePassthrough()} | doc_chain
    )

    response = retrievel_chain.invoke(input_prompt)
    st.write(response)