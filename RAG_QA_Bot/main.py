import streamlit as st
import os
import cohere
import PyPDF2
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# Create an instance of the Pinecone class
pc = Pinecone(api_key=PINECONE_API_KEY)

# Title of the app
st.title("PDF RAG Chatbot")

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

    # Embed the text content using Cohere
    client = cohere.Client(COHERE_API_KEY)
    embeddings = client.embed(model='large', texts=[text_content])
    embedding_vector = embeddings.embeddings[0]

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

    # Retrieve embeddings for the user's prompt
    user_embeddings = client.embed(model='large', texts=[prompt])
    user_embedding_vector = user_embeddings.embeddings[0]

    # Query Pinecone using the user's embedding
    query_results = index.query(vector=user_embedding_vector, top_k=3)  # Use keyword arguments

    # Generate a response based on retrieved content
    retrieved_texts = [item.id for item in query_results['matches']]
    context = "\n".join(retrieved_texts)

    # Generate response using Cohere's generate method
    prompt_with_context = """
    Extract the information from the pdf and answer the questions asked by the user
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    response = client.generate(
        model='command-r-plus-08-2024',  # Replace with a valid model name
        prompt=prompt_with_context,
        max_tokens=50,  # Adjust this based on your needs
        temperature=0.7,  # Adjust for creativity in responses
        stop_sequences=["User:"]
    ).generations[0].text.strip()

    # Append messages to session state
    st.session_state.messages.append({'role': 'User', 'message': prompt})
    st.session_state.messages.append({'role': 'Chatbot', 'message': response})

    st.chat_message("Chatbot").write(response)
