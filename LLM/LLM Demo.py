from langchain_openai import OpenAI # Inherated from BaseLLM
from dotenv import load_dotenv # Efficiently load and handle environment varaibles

load_dotenv()

llm = OpenAI(model='gpt-3.5-turbo-instruct')

result = llm.invoke("What is the capital of India")

print(result)