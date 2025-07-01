from langchain_groq import ChatGroq # Free Access
from dotenv import load_dotenv

def main():
    # Loading environment variables
    load_dotenv()

    # Creating an object of ChatGroq and provide the name of the model
    model = ChatGroq(model="llama-3.1-8b-instant", temperature=0.3, max_tokens=None)

    # Pass the query into the invoke function
    result = model.invoke("Who is the current prime minister of India?")

    # Print the result
    print(result.content)

if __name__ == "__main__":
    main()