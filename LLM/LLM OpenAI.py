from langchain_openai import OpenAI # Inherated from BaseLLM
from dotenv import load_dotenv # Efficiently load and handle environment varaibles

def main():
    # Loading environment variables
    load_dotenv()

    # Creating an object of OpenAI and provide the name of the model
    llm = OpenAI(model = 'gpt-3.5-turbo-instruct')

    # Pass the query into the invoke function
    result = llm.invoke("What is the capital of India")

    # Printing the response given by llm
    print(result) # The result is a plain text

if __name__ == "__main__":
    main()

# Dont use LLM's because they are old and cannot used for Conversactional Taksk.