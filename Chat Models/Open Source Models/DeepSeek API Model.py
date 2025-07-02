from tokenize import String
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

def main():
    # Loading environment variables
    load_dotenv()

    # Creating model by calling HF API
    llm = HuggingFaceEndpoint(
        repo_id="deepseek-ai/DeepSeek-R1-0528",
        task="text-generation",
        temperature=0.1,
        max_new_tokens=20
    )

    # Creating model object
    model = ChatHuggingFace(llm=llm)
    
    # Printing the result
    try:
        result = model.invoke("What is the capital of India?").content
        # DeepSeek model also provides thinking before the response
        result = result.split('</think>')[-1]
        print(result)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()