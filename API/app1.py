from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate
from langserve import add_routes
import uvicorn
from langchain_community.chat_models import ChatOllama
from dotenv import load_dotenv

load_dotenv()

#Server layer
app = FastAPI(
    title = "Langchain Server",
    version = "1.0",
    description = "A simple API Server"

)

## Ollama
llm = ChatOllama(model="llama3:latest")

#Prompts

prompt1 = ChatPromptTemplate.from_template("Write me an essay about {topic} with 50 words")
prompt2 = ChatPromptTemplate.from_template("Write me an poem about {topic} with 25 words")


#Routes 
add_routes(
    app,
    prompt1|llm,
    path="/essay"

)

add_routes(
    app,
    prompt2|llm,
    path="/poem"

)

if __name__ == "__main__":
    uvicorn.run(app,host="localhost", port=8000)
