import tempfile
import os
from llama_index import SimpleDirectoryReader, ServiceContext, set_global_service_context, VectorStoreIndex
from llama_index.query_engine import RetrieverQueryEngine
from llama_index.callbacks import CallbackManager, TokenCountingHandler
from llama_index.llms import OpenAI
import tiktoken
import openai

import streamlit as st

openai.api_key = st.secrets["OPENAI_API_KEY"]

st.set_page_config(layout = "wide")

st.title("Thothica PDF Retriever")

if "file_uploaded" not in st.session_state:
    st.session_state.file_uploaded = False

if "index" not in st.session_state:
    st.session_state.index = False

def file_uploaded():
    st.session_state.file_uploaded = True
    st.session_state.index = False

with st.expander(label = "File Upload", expanded = not st.session_state.file_uploaded):
    with st.form("file-upload"):
        files = st.file_uploader(label = "Please upload the PDF files -", accept_multiple_files = True)
        st.session_state.model_choice = st.selectbox("Which OpenAI model do you want to use?", options = ["gpt-4", "gpt-3.5-turbo", "gpt-4-1106-preview"], index = 1)
        similarity_top_k = st.slider(label = "Similarity top k", min_value = 2, max_value = 5, value = 2)
        submit = st.form_submit_button(label = "Submit", on_click = file_uploaded)

@st.cache_resource
def get_index(files):
    chatgpt = OpenAI(temperature = 0, model = st.session_state.model_choice)
    token_counter = TokenCountingHandler(tokenizer = tiktoken.encoding_for_model(st.session_state.model_choice).encode)
    callback_manager = CallbackManager([token_counter])
    service_context = ServiceContext.from_defaults(llm = chatgpt, callback_manager = callback_manager, chunk_size = 512)
    set_global_service_context(service_context)

    temp_dir = tempfile.TemporaryDirectory()
    for n, file in enumerate(files):
        file_path = os.path.join(temp_dir.name, file.name)
        with open(file_path, "wb") as f:
            f.write(file.read())
    docs = SimpleDirectoryReader(input_dir = temp_dir.name).load_data()
    index = VectorStoreIndex.from_documents(docs, show_progress = True)
    st.session_state.index = True
    st.write(f"{((token_counter.prompt_llm_token_count / 1000) * 0.01) + (token_counter.completion_llm_token_count / 1000) * 0.03} $ Used." if st.session_state.model_choice == "gpt-4" else f"{((token_counter.prompt_llm_token_count / 1000) * 0.001) + (token_counter.completion_llm_token_count / 1000) * 0.002} $ Used.")
    return index

if st.session_state.file_uploaded:
    index = get_index(files)
    if "messages" not in st.session_state:
        st.session_state.messages = []
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

if st.session_state.index:
    prompt = st.chat_input()
    if prompt:
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("assistant"):
            response = RetrieverQueryEngine(index.as_retriever(retriever_mode = 'embedding', similarity_top_k = similarity_top_k)).retrieve(prompt)
            text = ""
            for n, i in enumerate(response):
                text += f"## Node {n + 1}" + " \n\n**Text** - " + i.get_text() + " \n\n**Score** - " + str(i.get_score()) + " \n\n"
                for k in i.metadata.keys():
                    text += f" - **{k}**" +  " - " + str(i.metadata[k]) + " \n\n "
            st.write(text)
        st.session_state.messages.append({"role": "assistant", "content": text.strip()})