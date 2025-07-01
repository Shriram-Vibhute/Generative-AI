from langchain_anthropic import ChatAnthropic # Inherated from BaseChatModels
from dotenv import load_dotenv

def main():
    load_dotenv()
    model = ChatAnthropic(model='claude-3-5-sonnet-20241022')
    result = model.invoke('What is the capital of India') 
    print(result.content)

if __name__ == "__main__":
    main()