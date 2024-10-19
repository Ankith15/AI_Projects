import re
from io import BytesIO
from typing import Tuple, List
import cohere
from pypdf import PdfReader
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS

def pdf_parser(file: BytesIO) -> List[str]:
    pdf = PdfReader(file)
    output = [re.sub(r"\n\s*\n", "\n\n", re.sub(r"(?<!\n\s)\n(?!\s\n)", " ", re.sub(r"(\w+)-\n(\w+)", r"\1\2", page.extract_text().strip()))) for page in pdf.pages]
    return output

def text_to_docs(texts: List[str], filename: str) -> List[Document]:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, separators=["\n\n", "\n", ".", "!", "?", ",", " "], chunk_overlap=0)
    docs = [Document(page_content=chunk, metadata={"filename": filename, "chunk": i}) for i, chunk in enumerate(text_splitter.split_text("".join(texts)))]
    return docs

def doc_to_index(docs: List[Document], cohere_api_key: str) -> FAISS:
    co = cohere.Client(cohere_api_key)
    embeddings = co.embed(texts=[doc.page_content for doc in docs]).embeddings
    index = FAISS.from_documents(docs, embeddings)
    return index

def get_index_for_pdf(pdf_files: List[BytesIO], pdf_names: List[str], cohere_api_key: str) -> FAISS:
    docs = []
    for file, name in zip(pdf_files, pdf_names):
        text = pdf_parser(BytesIO(file))
        docs.extend(text_to_docs(text, name))
    return doc_to_index(docs, cohere_api_key)
