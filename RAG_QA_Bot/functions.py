import databutton as db
import re
from io import BytesIO
from typing import Tuple, List
import pickle
import cohere


from pypdf import PdfReader
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS


def pdf_parser(file: BytesIO,filename:str) -> Tuple[List[str],str]:
    pdf = PdfReader(file)
    output = []

    for page in pdf.pages:
        text = page.extract_text()
        text = re.sub(r"(\w+)-\n(\w+)",r"\1\2",text)
        text = re.sub(r"(?<!\n\s)\n(?!\s\n)"," ",text.strip())
        text = re.sub(r"\n\s*\n", "\n\n",text)
        output.append(text)
    return output, filename


def text_to_docs(text:List[str],filename:str) ->List[Document]:
    if isinstance(text,str):
        text = [text]

    page_docs = [Document(page_content=page) for page in text]
    for i, doc in enumerate(page_docs):
        doc.metadata['page'] = i + 1

    doc_chunk = []
    for doc in page_docs:
        text_splitter = RecursiveCharacterTextSplitter(
            Chunk_size = 4000,
            separators=["\n\n", "\n", ".","!","?", ",", " ", ""],
            chunk_overlap = 0
        )

        chunks= text_splitter.split_text(doc.page_content)
        for i, chunk in enumerate(chunks):
            doc = Document(
                page_content=chunk, metadata = {"page":doc.metadata['page'],"chunk":i}
            )
            doc.metadata['source'] = f"{doc.metadata['page']}-{doc.metadata["chunk"]}"
            doc.metadata['filename'] = filename
    return doc_chunk


def doc_to_index(docs, cohere_api_key):
    co = cohere.Client(cohere_api_key) 
    
    embeddings = co.embed(texts=[doc.page_content for doc in docs]).embeddings  
    
    index = FAISS.from_documents(docs, embeddings)  
    return index


def get_index_for_pdf(pdf_files, pdf_names, cohere_api_key):
    documents = []
    for pdf_file, pdf_name in zip(pdf_files, pdf_names):
        text, filename = pdf_parser(BytesIO(pdf_file), pdf_name)
        documents = documents + text_to_docs(text, filename)
    
    index = doc_to_index(documents, cohere_api_key) 
    return index

