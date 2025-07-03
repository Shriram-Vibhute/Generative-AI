import streamlit as st
import random
import time
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
load_dotenv()

# Streamed response emulator
def response_generator(input_text):
    # Creating model by calling HF API
    llm = HuggingFaceEndpoint(
        repo_id="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        task="text-generation",
        temperature=0.1,
        max_new_tokens=10
    )

    model = ChatHuggingFace(llm = llm)
    response = model.invoke(input_text).content
    response = response.split('</think>')[-1]

    for word in response.split():
        yield word + " "
        time.sleep(0.05)

st.title("Simple chat")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What is up?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        response = st.write_stream(response_generator(prompt))
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})