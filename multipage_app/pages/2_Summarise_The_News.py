import streamlit as st
from utils.rag import LangchainAssistant
from utils.vector_database import PineconeHelper

index_name = "all-news"

lc_assistant = LangchainAssistant(index_name=index_name)
pc_client = PineconeHelper(index_name=index_name)


def initialise_langchain():
    # Pinecone Index
    index = pc_client.pinecone_index()

    # Generate embeddings and setup vectorstore
    embeddings = lc_assistant.langchain_embeddings()
    vectorstore = lc_assistant.langchain_vectorstore(embeddings)

    # Model and retriever setup
    model = lc_assistant.langchain_model()
    retriever = lc_assistant.multi_query_retriever(vectorstore, model)

    # Prompt template setup
    prompt = lc_assistant.prompt_system_human_prompt()

    # Langchain chain setup
    chain = lc_assistant.langchain_chain(retriever, prompt, model)

    return chain


page_config = {
    "page_title": "Summarise The News",
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

if prompt := st.chat_input("Ask your question?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)

    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    chain = initialise_langchain()
    response = lc_assistant.langchain_streamlit_invoke(prompt=prompt, chain=chain)
    with st.chat_message("assistant"):
        response

    st.session_state.messages.append({"role": "assistant", "content": response})
