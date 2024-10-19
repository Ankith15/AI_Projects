import streamlit as st
from functions import get_index_for_pdf, CohereEmbeddings  # Ensure you have the correct imports
import os
from dotenv import load_dotenv
from langchain_community.llms import Cohere

# Load environment variables from .env file
load_dotenv()

st.title("RAG QA ChatBot")
COHERE_API_KEY = os.getenv("Cohere_api")

# Initialize Cohere embeddings using the custom class from functions.py
embedding_function = CohereEmbeddings(COHERE_API_KEY)

@st.cache_resource
def create_vectordb(files, filenames):
    # Use the embedding_function here correctly
    return get_index_for_pdf([file.getvalue() for file in files], filenames, COHERE_API_KEY)

# File upload
pdf_files = st.file_uploader("Upload PDF", type="pdf", accept_multiple_files=False)
if pdf_files:
    pdf_file_names = [pdf_files.name]
    st.session_state['vectordb'] = create_vectordb([pdf_files], pdf_file_names)

prompt_template = """ """  # Add template content here

# Display chat history
prompt = st.session_state.get("prompt", [{"role": "system", "content": "none"}])
for message in prompt:
    if message['role'] != "system":
        with st.chat_message(message['role']):
            st.write(message["content"])

# New question input
question = st.chat_input("Ask something new")
if question:
    vectordb = st.session_state.get('vectordb')
    if not vectordb:
        st.write("You need to provide a PDF")
        st.stop()

    # Search in vector database
    search_results = vectordb.similarity_search(question, k=3)
    pdf_extract = "\n".join([result.page_content for result in search_results])

    prompt[0] = {"role": "system", "content": prompt_template.format(pdf_extract=pdf_extract)}
    prompt.append({"role": "user", "content": question})

    # Generate response with Cohere
    with st.chat_message("assistant"):
        cohere_client = Cohere(COHERE_API_KEY)  # Using Cohere from langchain_community.llms
        cohere_response = cohere_client.generate(
            model='xlarge',
            prompt=prompt_template.format(pdf_extract=pdf_extract),
            max_tokens=200,
            temperature=0.5
        )
        result = cohere_response.generations[0].text.strip()
        st.write(result)

    # Update chat history
    prompt.append({"role": "assistant", "content": result})
    st.session_state["prompt"] = prompt
