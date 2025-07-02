from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
import os
# Requirement of PyTorch and CUDA

def main():
    # Storing model in D drive - Ingeneral, they were automatically saved in C drive
    os.environ['HF_HOME'] = 'D:/huggingface_cache'

    # Creating Model
    llm = HuggingFacePipeline.from_model_id(
        model_id='TinyLlama/TinyLlama-1.1B-Chat-v1.0', # 2.20GB
        task='text-generation',
        pipeline_kwargs=dict(
            temperature=0.5,
            max_new_tokens=100
        )
    )

    # Creating model object
    model = ChatHuggingFace(llm=llm)

    # Query to model
    result = model.invoke("What is the capital of India")

    # Printing result
    print(result.content)

if __name__ == '__main__':
    main()