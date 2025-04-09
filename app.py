import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import warnings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Suppress torch warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch")

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.error("Please set GROQ_API_KEY in your .env file")
    st.stop()

def get_pdf_text(pdf_docs):
    if not pdf_docs:
        return ""
    text = ""
    for pdf in pdf_docs:
        try:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                text += page.extract_text()
        except Exception as e:
            st.error(f"Error reading PDF: {str(e)}")
            return ""
    return text

def get_text_chunks(text):
    if not text:
        return []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        vector_store.save_local("faiss_index")
    except Exception as e:
        st.error(f"Error creating vector store: {str(e)}")
        return None

def get_conversational_chain():
    prompt_template = """
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    try:
        model = ChatGroq(
            api_key=GROQ_API_KEY,
            model_name="meta-llama/llama-4-scout-17b-16e-instruct",
            temperature=0.3
        )

        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
        return chain
    except Exception as e:
        st.error(f"Error initializing model: {str(e)}")
        return None

def user_input(user_question):
    try:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        if not os.path.exists("faiss_index"):
            st.error("Please upload and process PDF files first")
            return
        
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question)

        chain = get_conversational_chain()
        if chain is None:
            return
            
        response = chain(
            {"input_documents": docs, "question": user_question},
            return_only_outputs=True
        )

        st.write("Reply: ", response["output_text"])
    except Exception as e:
        st.error(f"Error processing your question: {str(e)}")

def main():
    st.set_page_config("Chat PDF")
    st.header("Chat with PDF using Llama💁")

    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                if raw_text:
                    text_chunks = get_text_chunks(raw_text)
                    if text_chunks:
                        get_vector_store(text_chunks)
                        st.success("Done")
                    else:
                        st.error("No text chunks generated from the PDF")
                else:
                    st.error("No text extracted from the PDF")

    # Add footer with team credits
    st.markdown("---")
    st.markdown("<h3 style='text-align: center;'>Team Debuggers</h3>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'><strong>Leader:</strong> <strong>Aamin Simmi Singh</strong></p>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Team Members: Sabreena Khan, Ankita Kumari, Poornima Sharma</p>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()