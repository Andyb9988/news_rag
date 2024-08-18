from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.prompts import (
    PromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.retrievers.multi_query import MultiQueryRetriever
import os

from logging_utils.log_helper import get_logger
from logging import Logger

logger: Logger = get_logger(__name__)

openai_api_key = os.getenv("OPENAI_API_KEY")
pinecone_api = os.getenv("PINECONE_API")


class LangchainAssistant:
    def __init__(self, index_name: str):
        self.pc_index = index_name

    def langchain_model(self):
        model = ChatOpenAI(temperature=0.2, api_key=openai_api_key, model="gpt-4o-mini")
        return model

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
        Use the following context to answer the question. Please summarize the key points in a polite, concise, and clear manner. 
        Ensure the summary captures the most important information, highlights any critical details, and is easy to understand.

        Context: {context}
        Question: {question}

        Summary:
        """
        return PromptTemplate(
            input_variables=["context", "question"], template=prompt_template
        )

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