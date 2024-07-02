import streamlit as st
from langchain_helper import get_qa_chain, create_vector_db

st.title(" RAG application Transcribing PDF Q&A ")
#uploaded_file = st.file_uploader("Select a PDF file:", type=["pdf"])

btn = st.button("Create Knowledgebase")
if btn:
    create_vector_db()

question = st.text_input("Question: ")

if question:
    chain = get_qa_chain()
    response = chain(question)

    st.header("Answer")
    st.write(response["result"])
    
    