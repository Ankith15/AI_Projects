# PDF RAG Chatbot
##Overview
This project implements a Retrieval-Augmented Generation (RAG) chatbot that allows users to upload a PDF file, ask questions based on the PDF content, and receive coherent, contextually relevant answers. The system integrates Cohere API for generating answers and Pinecone for efficient document embedding storage and retrieval.

## Features
### PDF Upload: Users can upload PDF files.
### Text Extraction: The system extracts text from the uploaded PDF using PyPDF2.
### Embedding Creation: The extracted text is embedded using Cohere's language model.
### Embedding Storage: The embeddings are stored in a Pinecone vector database for efficient retrieval.
### Question Answering: Users can input questions, and the chatbot retrieves the relevant sections of the document and generates an appropriate answer using Cohere's text generation model.
### Interactive Interface: Built using Streamlit, the chatbot provides a user-friendly interface for real-time interaction.
## Technologies Used
### Streamlit: Frontend for the interactive chatbot interface.
### Cohere API: Used to generate embeddings and answer questions based on document content.
### Pinecone: Vector database for storing and retrieving document embeddings.
### PyPDF2: PDF text extraction.
### Dotenv: Environment variable management.

## Usage
### Upload a PDF: Upload a PDF file using the file uploader in the chatbot interface.
### Ask Questions: After the PDF is processed, type your question related to the PDF content in the input box.
### Receive Answers: The chatbot retrieves the relevant sections from the document, and generates an answer using Cohere's language model.

## Key Components
### 1. PDF Processing:
The uploaded PDF is parsed, and text is extracted using PyPDF2.
### 2. Embedding Creation:
The extracted text is sent to Cohere's embedding model (large) to create a vector representation of the text.
### 3. Pinecone Vector Database:
If the index for the PDF does not already exist, a new one is created. The document's embeddings are stored in Pinecone, allowing for quick and efficient search and retrieval.
### 4. Question Answering:
When a question is asked, its embedding is compared with the stored embeddings in Pinecone to retrieve the most relevant text segments from the document.
A prompt with the retrieved content is sent to Cohere’s text generation model to generate a concise, relevant answer.