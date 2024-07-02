from langchain.vectorstores import FAISS
from langchain.llms import GooglePalm
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader
import os

from dotenv import load_dotenv
load_dotenv() 
# take environment variables from .env (especially openai api key)
api_key='AIzaSyAya3JqYwE3f1v88tHIluGnKse114wJdEg'
llm = GooglePalm(google_api_key=api_key, temperature=0.1)

instructor_embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-large")
vectordb_file_path = "faiss_index"

def create_vector_db():
    #loader = PyMuPDFLoader("F:/LLM project/RAG+LLM/project/Diksha Pandey Resume.pdf",source_column="prompt")
    loader = PyPDFLoader("F:\LLM project\RAG+LLM\project\Diksha Pandey Resume.pdf")
    pages = loader.load_and_split()
    pages = loader.load()
    
    
    # Create a FAISS instance for vector database from 'data'
    vectordb = FAISS.from_documents(documents=pages,
                                    embedding=instructor_embeddings)
 # Save vector database locally
    vectordb.save_local(vectordb_file_path)


def get_qa_chain():
    # Load the vector database from the local folder
    vectordb = FAISS.load_local(vectordb_file_path, instructor_embeddings)

    # Create a retriever for querying the vector database
    retriever = vectordb.as_retriever(score_threshold=0.7)

    prompt_template = """Given the following context and a question, generate an answer based on this context only.
    In the answer try to provide as much text as possible from "response" section in the source document context without making much changes.
    If the answer is not found in the context, kindly state "I don't know." Don't try to make up an answer.

    CONTEXT: {context}

    QUESTION: {question}"""

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    chain = RetrievalQA.from_chain_type(llm=llm,
                                        chain_type="stuff",
                                        retriever=retriever,
                                        input_key="query",
                                        return_source_documents=True,
                                        chain_type_kwargs={"prompt": PROMPT})

    return chain

if __name__ == "__main__":
    create_vector_db()
    chain = get_qa_chain()
    print(chain("Do you have javascript course?"))