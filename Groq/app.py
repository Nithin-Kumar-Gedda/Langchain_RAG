import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS

from dotenv import load_dotenv
load_dotenv()

### Load the GROQ API Key
groq_api_key = os.environ['GROQ_API_KEY']

if "vector" not in st.session_state:
    st.session_state.embeddings = OllamaEmbeddings(model="llama3.2")
    st.session_state.loader = WebBaseLoader("https://docs.smith.langchain.com/")
    st.session_state.docs = st.session_state.loader.load()
    st.session_state.txt_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    st.session_state.final_docs = st.session_state.txt_splitter.split_documents(st.session_state.docs[:50])
    st.session_state.vectors = FAISS.from_documents(st.session_state.final_docs,st.session_state.embeddings)



st.title("ChatGROQ Demo😊")
llm = ChatGroq(model="llama-3.3-70b-versatile", groq_api_key=groq_api_key)

prompt = ChatPromptTemplate.from_template(
"""
 Answer the some questions based on the provided comntext only.
 Please provide the most accurate response based on the question.
 <context>
 {context}
 </context>
 Question:{input}
 
 """
)

doc_chain = prompt | llm | StrOutputParser() # inplace of create_stuff_documents_chain
retriever = st.session_state.vectors.as_retriever()

retrievel_chain = (
    {"context": retriever, "input": RunnablePassthrough()} | doc_chain
)

user_prompt = st.text_input("Input your prompt here!!!")

if prompt:
    response = retrievel_chain.invoke(user_prompt)
    st.write(response)

    with st.expander("Document Similarity Search"):
        docs = retriever.invoke(user_prompt)
        for i, doc in enumerate(docs):
            st.write(doc.page_content)
            st.write("-----------------------------")







