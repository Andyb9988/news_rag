from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain.prompts import (
    PromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
)
from langchain.retrievers.multi_query import MultiQueryRetriever
import streamlit as st
import logging
from logging_utils.log_helper import get_logger
from logging import Logger
import os
# logging.basicConfig(level=logging.DEBUG, filename='app.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')
logger: Logger = get_logger(__name__)
openai_api_key = os.getenv("OPENAI_API_KEY")

class LangchainAssistant:
    def __init__(self, index_name: str):
        self.pc_index = index_name

    def langchain_embeddings(self, model_name="text-embedding-3-small"):
        embeddings = OpenAIEmbeddings(model=model_name, dimensions=1536)
        return embeddings

    def langchain_vectorstore(self, embeddings):
        vectorstore = PineconeVectorStore(self.pc_index, embeddings)
        logger.info(
            f"initialised vectore store: {vectorstore} using index: {self.pc_index}"
        )
        return vectorstore

    def langchain_retriever(self, vectorstore):
        retriever = vectorstore.as_retriever()
        return retriever

    def multi_query_retriever(self, vectorstore, llm):
        vectorstore_retriever = vectorstore.as_retriever(k=5)
        logger.info(f"initialised retriever: {vectorstore_retriever}")
        retriever = MultiQueryRetriever.from_llm(
            retriever=vectorstore_retriever, llm=llm
        )
        logger.info(f"The retrieved content: {retriever}")
        return retriever

    def summarise_prompt(self):
        prompt_template = """
        Please summarize the key points of the article in a polite, concise, and clear manner./n 
        Ensure the summary captures the most important information, highlights any critical details,/n 
        and is easy to understand.
        context: {context}
        qustion: {question}

        Summary:

        """
        prompt = PromptTemplate(
            input_variables = ["context", "question"],
            template = prompt_template
        )
        return prompt


    def prompt_system_human_prompt(self):
        review_system_template_str = """
        You are a highly efficient virtual assistant 
        designed to answers user queries 
        through extract valuable information 
        from the context provided. 
        Your primary goal is to provide insightful and concise summaries of the content within the transcripts. 
        You excel in identifying key topics, extracting relevant details, and presenting the information in a clear and coherent manner. 
        Your users rely on you to distill complex video content into easily understandable insights. 
        Keep in mind the importance of accuracy, clarity, and brevity in your responses.
        If the question cannot be answered using the information provided say "I don't know".                
        context:
        {context}
        """

        review_system_prompt = SystemMessagePromptTemplate(
            prompt=PromptTemplate(
                input_variables=["context"], template=review_system_template_str
            )
        )

        review_human_prompt = HumanMessagePromptTemplate(
            prompt=PromptTemplate(input_variables=["question"], template="{question}")
        )

        messages = [review_system_prompt, review_human_prompt]
        review_prompt_template = ChatPromptTemplate(
            input_variables=["context", "question"],
            messages=messages,
        )

        return review_prompt_template

    def langchain_model(self):
        model = ChatOpenAI(temperature=0.2, api_key = openai_api_key, model="gpt-4o-mini")
        return model

    def langchain_chain(self, retriever, prompt, model):
        chain = (
            RunnableParallel({"context": retriever, "question": RunnablePassthrough()})
            | prompt
            | model
            | StrOutputParser()
        )
        logging.info("Langchain chain running correctly.")
        return chain

    def langchain_invoke(self, chain):
        query = str(input("What do you want to search on Youtube?"))
        try:
            result = chain.invoke(query)
            print(result)
        except Exception as e:
            print(f"Error: {str(e)}")

    def langchain_streamlit_invoke(self, prompt, chain):
        query = prompt
        try:
            result = chain.invoke(query)
            if result is None:
                st.write(
                    "No response generated."
                )  # Handle None by displaying a default message
                logging.warning("No responses were generated by the model.")
            else:
                st.write(result)  # Use st.write for string outputs
                logging.info("Response generated successfully")
        except Exception as e:
            st.write(f"Error: {str(e)}")
