from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

def main() -> None:
    # Loading environment variables
    load_dotenv()

    # Creating an object of OpenAI and provide the name of the model
    model = ChatOpenAI(model = "gpt-4", temperature = 0.3, max_completion_tokens = 10)
    # temperature: Creativity patameter - It affects how creative or deterministic the responses are.

    # Pass the query into the invoke function
    result = model.invoke(input = "What is the capital of india")

    # Printing the response given by llm
    print(result.content) # The resule which you get is not just a plain text, it actually provide metadata along with the response.

if __name__ == "__main__":
    main()