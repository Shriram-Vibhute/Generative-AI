# Importing necessary libraries
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import load_prompt # Loading the prompt saved in json file
from dotenv import load_dotenv

def model_building():
    # Creating model by calling HF API
    llm = HuggingFaceEndpoint(
        repo_id="deepseek-ai/DeepSeek-R1-0528",
        task="text-generation",
        temperature=0.1,
        max_new_tokens=20
    )

    model = ChatHuggingFace(llm = llm)
    return model

def prompt_loader():
    # Loading prompt
    template = load_prompt('template.json')
    return template

def website_ui():
    print("Available papers:")
    print("1. Attention Is All You Need")
    print("2. BERT: Pre-training of Deep Bidirectional Transformers")
    print("3. GPT-3: Language Models are Few-Shot Learners")
    paper_input = input("Enter Paper Number (1-3): ")
    
    # Convert input to paper name
    paper_names = {
        "1": "Attention Is All You Need",
        "2": "BERT: Pre-training of Deep Bidirectional Transformers",
        "3": "GPT-3: Language Models are Few-Shot Learners"
    }
    paper_input = paper_names.get(paper_input, "Invalid paper selection")
    style_input = input("Select Explanation Style (Beginner-Friendly/Technical/Code-Oriented/Mathematical): ")
    length_input = input("Select Explanation Length (Short/Medium/Long): ")

    model = model_building()
    template = prompt_loader()
    prompt = template.invoke({'paper_input': paper_input, 'style_input': style_input, 'length_input': length_input})
    result = model.invoke(prompt)
    print(result.content)

def main():
    load_dotenv()
    website_ui()

if __name__ == "__main__":
    main()