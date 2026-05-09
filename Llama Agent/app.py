import streamlit as st # Interface platform
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter  # Text splitter
from langchain_community.vectorstores import FAISS   # embeddings store
from langchain_ollama.embeddings import OllamaEmbeddings  # text to embeddings
from langchain_groq import ChatGroq    # access groq hosted LLM
from langchain_core.output_parsers import StrOutputParser 
from langchain_core.runnables import RunnablePassthrough
from langchain_community.document_loaders import PyPDFDirectoryLoader  # reads the PDF files
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()
# load the GROQ API Key
groq_api_key = os.environ['GROQ_API_KEY']

st.title("ChatGROQ with Llama3 Demo")

llm = ChatGroq(groq_api_key=groq_api_key,
               model="llama-3.1-8b-instant")

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

def vector_embedding():

    if "vectors" not in st.session_state:

        st.session_state.embeddings = OllamaEmbeddings(model="llama3:latest")
        st.session_state.loader = PyPDFDirectoryLoader("./data") # data ingestion
        st.session_state.docs = st.session_state.loader.load() # document loading
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200) #chunk creation
        st.session_state.final_docs = st.session_state.text_splitter.split_documents(st.session_state.docs[:20]) # splitting
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_docs, st.session_state.embeddings) # vectors Ollama embeddings


prompt1 = st.text_input("Enter your question from documents")

if st.button("Document Embedding"):
    vector_embedding()
    st.write("Vector score database is ready!!!")


if prompt1:
    if "vectors" not in st.session_state:
        st.warning("Please click on the 'Document Embeddings' button first")
    else:
        document_chain = prompt | llm | StrOutputParser()
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = (
                {"context": retriever, "input":RunnablePassthrough()} | document_chain
            )
        response=retrieval_chain.invoke(prompt1)
        st.write(response)

        with st.expander("Document Similarity Search"):
            docs = retriever.invoke(prompt1)
            for i, doc in enumerate(docs):
                st.write(doc.page_content)
                st.write("-----------------------------")
