import requests
import streamlit as st

#essay API
def get_essay_response(input_text):
    response = requests.post("http://localhost:8000/essay/invoke",
    json ={'input':{'topic':input_text}})

    return response.json()['output']['content']

#poem API
def get_poem_response(input_text):
    response = requests.post("http://localhost:8000/poem/invoke",
    json = {'input':{'topic':input_text}})

    return response.json()['output']['content']


## streamlit framework

st.title("Langchaim Demo with LLAMA3 API")

essay_input = st.text_input("Write an essay on")
if essay_input:
    essay_result = get_essay_response(essay_input)
    st.write(essay_result)

poem_input = st.text_input("Write a peom on")
if poem_input:
    poem_result = get_poem_response(poem_input)
    st.write(poem_result)