from langchain_core.prompts import ChatPromptTemplate
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

# In recent version
chat_template = ChatPromptTemplate([
    ('system', 'You are a helpful {domain} expert'),
    ('human', 'Explain in simple terms, what is {topic}')
])

# This will not work - The parameters values will not shown in the prompts
'''
    chat_template = ChatPromptTemplate([
        SystemMessage(content = ('You are a helpful {domain} expert')),
        HumanMessage(content = ('Explain in simple terms, what is {topic}'))
    ])
'''
prompt = chat_template.invoke({'domain':'cricket','topic':'Bat'})
response = model.invoke(prompt).content
print(response)