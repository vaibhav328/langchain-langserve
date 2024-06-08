# app/server.py
from dotenv import load_dotenv
from fastapi import FastAPI
from langchain_cohere import ChatCohere
from langchain.prompts import ChatPromptTemplate
from langserve import add_routes
import os

load_dotenv()
app = FastAPI()

model = ChatCohere(cohere_api_key = os.getenv("COHERE_API_KEY"))
prompt = ChatPromptTemplate.from_template("Give me a summary about {topic} in a paragraph or less.")
chain = prompt | model

add_routes(app, chain, path="/cohere")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)