import streamlit as st
import os
import google.generativeai as gemini
import PyPDF2
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# Set up Google Gemini API key
gemini.configure(api_key=GOOGLE_API_KEY)

# Create an instance of the Pinecone class
pc = Pinecone(api_key=PINECONE_API_KEY)

# Title of the app
st.title("PDF RAG Chatbot (Using Google Gemini)")

# Initialize session state for messages
if "messages" not in st.session_state:
    st.session_state["messages"] = [{'role': "chatbot", 'message': "Upload a PDF and ask a question!"}]

# Display chat messages
for msg in st.session_state.messages:
    st.chat_message(msg['role']).write(msg['message'])

# PDF upload section
pdf_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if pdf_file:
    # Extract text from the uploaded PDF
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text_content = ""
    for page in pdf_reader.pages:
        text_content += page.extract_text() + "\n"

    # Generate embeddings for the PDF text
    # Make sure to replace `embed_text` with the actual method name for generating embeddings
    embedding_response = gemini.embed_text(text_content)  # Adjust to the correct method name
    embedding_vector = embedding_response.embedding  # Change according to the actual response structure

    index_name = "pdf-embeddings"

    # Check if the index exists; if not, create it
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=len(embedding_vector),  # Use the length of the embedding vector
            metric='euclidean',
            spec=ServerlessSpec(cloud='aws', region='us-east-1')  # Use a supported region
        )

    # Upsert the embedding into Pinecone
    index = pc.Index(index_name)
    index.upsert([(pdf_file.name, embedding_vector)])  # Use the filename as the ID

    st.success("PDF processed and vectors stored!")

# User input for questions
prompt = st.chat_input("Ask a question about the PDF...")

if prompt:
    st.chat_message("User").write(prompt)

    # Generate embeddings for the user prompt
    # Again, replace `embed_text` with the actual method name for generating embeddings
    user_embedding_response = gemini.embed_text(prompt)  # Adjust to the correct method name
    user_embedding_vector = user_embedding_response.embedding  # Change according to the actual response structure

    # Query Pinecone using the user's embedding
    query_results = index.query(vector=user_embedding_vector, top_k=3)

    # Generate a response based on retrieved content
    retrieved_texts = [item.id for item in query_results['matches']]
    context = "\n".join(retrieved_texts)

    # Prepare the prompt for Gemini's text generation
    prompt_with_context = f"""
    Extract the information from the PDF and answer the questions asked by the user.
    Context:\n{context}\n
    Question:\n{prompt}\n

    Answer:
    """

    # Generate a response using the correct text generation method
    # Adjust to the correct method name in the Gemini API
    response = gemini.generate_text(prompt_with_context)  # Change to the correct method name
    generated_text = response.text.strip()  # Adjust to access the generated text correctly

    # Append messages to session state
    st.session_state.messages.append({'role': 'User', 'message': prompt})
    st.session_state.messages.append({'role': 'Chatbot', 'message': generated_text})

    st.chat_message("Chatbot").write(generated_text)
