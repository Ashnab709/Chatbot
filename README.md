import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import os

# Streamlit UI
st.header("📄 PDF Chatbot (Local Hugging Face Model)")

with st.sidebar:
    st.title("📁 Upload Documents")
    file = st.file_uploader("Upload PDF and ask questions", type="pdf")

if file is not None:
    # Read and extract PDF text
    pdf_reader = PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text

    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n"],
        chunk_size=1000,
        chunk_overlap=150,
        length_function=len
    )
    chunks = text_splitter.split_text(text)

    # Generate embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Create vector store
    vector_store = FAISS.from_texts(chunks, embeddings)

    # Get user input
    user_question = st.text_input("💬 Ask a question about the document:")

    if user_question:
        # Perform similarity search
        matched_docs = vector_store.similarity_search(user_question)

        # Load local model (you can change the model)
        model_id = "mistralai/Mistral-7B-Instruct-v0.1"

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto", device_map="auto")
        pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=512)
        llm = HuggingFacePipeline(pipeline=pipe)

        # Load QA chain
        chain = load_qa_chain(llm, chain_type="stuff")

        # Run QA chain
        response = chain.run(input_documents=matched_docs, question=user_question)

        # Display response
        st.markdown("### 🤖 Answer:")
        st.write(response)
