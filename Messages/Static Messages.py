from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    task="text-generation",
    temperature=0.1,
    max_new_tokens=10
)

model = ChatHuggingFace(llm = llm)

messages = [
    SystemMessage(content = 'You are an experienced AI engineer'),
    HumanMessage(content = 'Tell me about LangChain')
]

result = model.invoke(messages)
messages.append(AIMessage(content = result.content))

print(messages)