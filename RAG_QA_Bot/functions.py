import re
from io import BytesIO
from typing import List
import cohere
from pypdf import PdfReader
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.faiss import FAISS

# Function to extract text from PDF
def pdf_parser(file: BytesIO) -> List[str]:
    pdf = PdfReader(file)
    output = [re.sub(r"\n\s*\n", "\n\n", re.sub(r"(?<!\n\s)\n(?!\s\n)", " ", re.sub(r"(\w+)-\n(\w+)", r"\1\2", page.extract_text().strip()))) for page in pdf.pages]
    return output

# Function to split text into smaller chunks
def text_to_docs(texts: List[str], filename: str) -> List[Document]:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=0)
    docs = [Document(page_content=chunk, metadata={"filename": filename, "chunk": i}) for i, chunk in enumerate(text_splitter.split_text("".join(texts)))]
    return docs

# Custom Cohere embedding class
class CohereEmbeddings:
    def __init__(self, COHERE_API_KEY: str):
        self.client = cohere.Client(COHERE_API_KEY)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        response = self.client.embed(texts=texts)
        return response.embeddings

# Function to create FAISS index from documents
def doc_to_index(docs: List[Document], cohere_embed: CohereEmbeddings) -> FAISS:
    index = FAISS.from_documents(docs, cohere_embed)
    return index

# Main function to get FAISS index from PDFs
def get_index_for_pdf(pdf_files: List[BytesIO], pdf_names: List[str], cohere_embed: CohereEmbeddings) -> FAISS:
    docs = []
    for file, name in zip(pdf_files, pdf_names):
        text = pdf_parser(BytesIO(file))
        docs.extend(text_to_docs(text, name))
    return doc_to_index(docs, cohere_embed)
