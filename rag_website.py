# import sys
# print("Python executable being used:", sys.executable)

import os
import streamlit as st
from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain

from dotenv import load_dotenv
load_dotenv()

# load api key
os.environ["NVIDIA_API_KEY"] = os.getenv("NVIDIA_API_KEY")
# print(os.environ["NVIDIA_API_KEY"])


def upload_data():
    # load
    loader = PyPDFDirectoryLoader("research_papers")
    documents = loader.load()

    # split using recursive --functionality of recursive is to split the documents semantically
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 500, chunk_overlap = 50, separators=None
    )

    splitted_documents = text_splitter.split_documents(documents=documents)
    return splitted_documents


def create_vector_store(splitted_documents):
    embeddings = NVIDIAEmbeddings()
    vector_DB = FAISS.from_documents(splitted_documents, embeddings)
    vector_DB.save_local("faiss_index")


def create_llm_model():
    llm = ChatNVIDIA(model="meta/llama3-70b-instruct")
    # llm = ChatNVIDIA(model="meta/llama3-8b-instruct")
    return llm


def get_response(llm, vector_DB, question, chat_history):
    # initialize a conversational retrieval chain,  
    # #####  explore chain_type="stuff", search_type = "similarity", search_kwargs = {"k":3}, chain_type_kwargs={"prompt":PROMPT}
    query = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever = vector_DB.as_retriever(),
        return_source_documents = True
    )

    # invoke the chain
    response = query.invoke({"question": question, "chat_history": chat_history})
    return response





















