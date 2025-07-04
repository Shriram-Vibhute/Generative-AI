from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

def main():
    load_dotenv()
    model = ChatGoogleGenerativeAI(model='gemini-2.0-flash')
    result = model.invoke('What is the capital of India')
    print(result.content)

if __name__ == "__main__":
    main()