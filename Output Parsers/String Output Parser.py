# Importing necessary libraries
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

# Loading environment variables
load_dotenv()

# Building the LLM model
llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-R1-0528",
    task="text-generation",
    temperature=0.1,
    max_new_tokens=20
)

# Creating model object
model = ChatHuggingFace(llm=llm)

# 1st prompt -> detailed report
template1 = PromptTemplate(
    template='Write a detailed report on {topic}',
    input_variables=['topic']
)

# 2nd prompt -> summary
template2 = PromptTemplate(
    template='Write a 5 line summary on the following text. /n {text}',
    input_variables=['text']
)

# Creating output parser object
parser = StrOutputParser()

# Creating chain
chain = template1 | model | parser | template2 | model | parser

# Executing chain
result = chain.invoke({'topic':'black hole'})

# Printing the result.
print(result)