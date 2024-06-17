import os
import re

# Set the environment variable to handle OpenMP duplicate libraries issue
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import streamlit as st
from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import FAISS
from rag_website import *
from yolo_inference import detect_disease
from pfmea import perform_pfmea
from PIL import Image, UnidentifiedImageError
from utils import filter_potential_causes

from dotenv import load_dotenv
load_dotenv()

# Load API key
os.environ["NVIDIA_API_KEY"] = os.getenv("NVIDIA_API_KEY")

############

# Directory where uploaded files will be saved
UPLOAD_DIR = "/Users/dheeraj/Desktop/NVIDIA_GENAI_CONTEST/research_papers"

# Ensure the UPLOAD_DIR directory exists
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

def handle_uploaded_file(uploaded_file):
    # Function to handle the uploaded file
    file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success(f"File {uploaded_file.name} uploaded successfully.")
############

def main():
    st.set_page_config("GenAI_Agent")
    st.title("AI-Driven Agri-Agent for Risk Prevention and Root Cause Analysis using NVIDIA NIM 🤖")
    st.image("img/main_chat_logo.png")

    st.markdown(
        """
        <style>
        .smaller-font {
            font-size:17px !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown('<p class="smaller-font">Unlock the combined potential of YOLOv8 and Operation Research with the Llama-3 LLM model using Retrieval Augmented Generation(RAG). This intelligent agent detects plant status, analyzes feedback, stores it, identifies potential and root causes, and effectively responds to user queries with a knowledge base. Enhance your model with regular updates.</p>', unsafe_allow_html=True)

    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.session_state.disease_info = {}
        st.session_state.potential_causes = []
        st.session_state.root_cause = None

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


    with st.sidebar:
        st.title("Upload Documents")

        uploaded_file = st.file_uploader("Upload Documents", type=["pdf", "docx", "txt"])
        if uploaded_file:
            handle_uploaded_file(uploaded_file)

    with st.sidebar:
        st.title("Vector database")
        st.image("img/faiss_logo.png")

        if st.button("Click to update it"):
            with st.spinner("Processing..."):
                docs = upload_data()
                create_vector_store(docs)
                st.success("Done")

        st.title("Yolo Prediction")
        st.image("img/yolo.png")

        uploaded_image = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
        if uploaded_image:
            with st.spinner("Processing..."):
                disease = detect_disease(uploaded_image)
                st.session_state.disease = disease
                st.session_state.disease_info[disease] = {"disease": disease}
                st.success(f"Disease detected: {disease}")

        if "disease" in st.session_state:
            if "user_feedback" not in st.session_state:
                st.session_state.user_feedback = ""

            user_feedback = st.text_area("Provide your comments or feedback on the detected disease", value=st.session_state.user_feedback, key="user_feedback_input")

            if st.button("Submit Feedback"):
                if not user_feedback.strip():  # Check if user_feedback is empty
                    st.error("Please provide your feedback before submitting.")
                else:
                    with st.spinner("Processing..."):
                        st.session_state.user_feedback = user_feedback

                        llm = create_llm_model()
                        embeddings = NVIDIAEmbeddings()
                        faiss_index = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

                        predefined_causes = [
                            "fungi", "over watering", "high humidity", "poor air circulation",
                            "insufficient sunlight", "nutrient deficiency", "insect infestation",
                            "soil pH imbalance", "contaminated tools", "poor soil drainage"
                        ]

                        prompt = (
                            f"The detected disease is {st.session_state.disease}. "
                            f"User's feedback: {st.session_state.user_feedback}. "
                            f"Considering the following predefined potential causes: {', '.join(predefined_causes)}. "
                            f"What are the most likely causes for the detected disease based on provided documents and predefined causes?"
                        )

                        response = get_response(llm=llm, vector_DB=faiss_index, question=prompt, chat_history=st.session_state.chat_history)
                        
                        response_text = response["answer"]
                        print("response_for potential cause_from_llm ::::::::::::", response_text)
                        st.session_state.potential_causes = filter_potential_causes(response_text, predefined_causes)
                        st.session_state.disease_info[st.session_state.disease]["potential_causes"] = st.session_state.potential_causes
                        st.success(f"Potential causes: {', '.join(st.session_state.potential_causes)}")

                        # Store the combined context in memory
                        st.session_state.chat_history.append((
                            f"Disease: {st.session_state.disease}",
                            f"User Feedback: {st.session_state.user_feedback}",
                            f"Potential Causes: {', '.join(st.session_state.potential_causes)}"
                        ))

        # Always display potential causes if they exist
        if st.session_state.potential_causes:
            st.subheader("Potential Causes:")
            for cause in st.session_state.potential_causes:
                st.markdown(f"- {cause}")

        st.title("PFMEA")
        st.image("img/operation_research.png")

        if st.button("Click to get root cause"):
            with st.spinner("Processing..."):
                if "potential_causes" in st.session_state and st.session_state.potential_causes:
                    print("potential_causes::::::", st.session_state.potential_causes)
                    root_cause = perform_pfmea(st.session_state.potential_causes)
                    st.session_state.root_cause = root_cause
                    st.session_state.disease_info[st.session_state.disease]["root_cause"] = root_cause
                    st.success(f"Root Cause: {root_cause}")

                    # Store the combined context in memory
                    st.session_state.chat_history.append((
                        f"Disease: {st.session_state.disease}",
                        f"Root Cause: {root_cause}",
                        f"Potential Causes: {', '.join(st.session_state.potential_causes)}"
                    ))

        # Always display the root cause if it exists
        if st.session_state.root_cause:
            st.subheader(f"Root Cause: {st.session_state.root_cause}")

    if prompt := st.chat_input("Enter your query here: "):
        with st.spinner("Working on your query..."):
            llm = create_llm_model()
            embeddings = NVIDIAEmbeddings()
            faiss_index = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

            response = get_response(llm=llm, vector_DB=faiss_index, question=prompt, chat_history=st.session_state.chat_history)
            with st.chat_message("user"):
                st.markdown(prompt)
            with st.chat_message("assistant"):
                st.markdown(response["answer"])

            st.session_state.messages.append({"role": "user", "content": prompt})
            st.session_state.messages.append({"role": "assistant", "content": response["answer"]})
            st.session_state.chat_history.extend([(prompt, response["answer"])])

            if "root_cause" in st.session_state and "remedies" not in st.session_state:
                st.session_state.remedies = response["answer"]
                st.sidebar.success(f"Remedies: {response['answer']}")

            # Check if the question relates to disease, causes, or root cause
            for disease, info in st.session_state.disease_info.items():
                if disease in prompt:
                    st.session_state.messages.append({"role": "assistant", "content": f"The recent root cause for {disease} was {info['root_cause']} due to {', '.join(info['potential_causes'])}"})
                    st.sidebar.success(f"The recent root cause for {disease} was {info['root_cause']} due to {', '.join(info['potential_causes'])}")
                    break

if __name__ == "__main__":
    main()
