import streamlit as st
from streamlit_message import message
from utils.rag import YoutubeSearchAssistant
from utils.helper import Helper
from utils.vector_database import PineconeHelper

config_path = "multipage_app/utils/config.json"
helper = Helper(config_path)
config = helper.load_config()
pinecone_api = config["PINECONE_API_KEY"]
openai_api = config["OPENAI_API_KEY"]

yt_search = YoutubeSearchAssistant()
pc = PineconeHelper(pinecone_api, index_name="golf")


def langchain():
    # Pinecone
    index = pc.pinecone_index()
    embeddings = yt_search.langchain_embeddings()
    vectorstore = yt_search.langchain_vectorstore(index, embeddings)
    model = yt_search.langchain_model()

    retriever = yt_search.multi_query_retriever(vectorstore, model)
    prompt = yt_search.prompt_system_human_prompt()

    chain = yt_search.langchain_chain(retriever, prompt, model)

    return chain


page_config = {
    "page_title": "Ask Youtube Anything",
    "layout": "centered",
    "initial_sidebar_state": "auto",
}
st.set_page_config(**page_config)
st.markdown(
    "<h1 style='text-align: center;'> Ask Youtube Anything </h1>",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown(
        """
        <h2> Welcome to a simple RAG APP </h2>
        """,
        unsafe_allow_html=True,
    )


def clear_chat_history():
    st.session_state.messages = [
        {"role": "assistant", "content": "How may I assist you today?"}
    ]


if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "system", "content": "You are a helpful assistant."}
    ]

st.sidebar.button("Clear Chat History", on_click=clear_chat_history)

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

response = ""

if prompt := st.chat_input("Say something?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)

    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    chain = langchain()
    response = yt_search.langchain_streamlit_invoke(prompt=prompt, chain=chain)
    with st.chat_message("assistant"):
        response

    st.session_state.messages.append({"role": "assistant", "content": response})
