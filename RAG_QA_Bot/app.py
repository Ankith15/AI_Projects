import databutton  as db
import streamlit as st
import cohere
from functions import get_index_for_pdf
from langchain.chains import RetrivalAQ
from langchain.llms import cohere
import os
from dotenv import load_dotenv
from langchain.llms import Cohere as LangchainCohere
import cohere
from langchain.llms import Cohere as LangchainCohere
import cohere

load_dotenv()

st.title("RAG QA ChatBot")

os.environ['Cohere_api'] = db.secrets.get("Cohere_api")
cohere.api_key = db.secrets.get("Cohere_api")


@st.cache_data

def create_vectordb(files,filenames):
    with st.spinner("Vector database"):
            vectordb = get_index_for_pdf(
                  [file.getvalue() for file in files], filenames, cohere.api_key       
            )
            return vectordb

pdf_files = st.file_uploader("",type = "pdf",accept_multiple_files=False)
if pdf_files:
      pdf_file_names = [file.name for file in pdf_files]
      st.session_state['vectordb'] = create_vectordb(pdf_files,pdf_file_names)


prompt_template = """


"""

prompt = st.session_state.get("prompt",[{"role":"system","content":"none"}])

for message in prompt:
      if message['role'] != "system":
            with st.chat_message(message['role']):
                  st.write(message["content"])

question = st.chat_input("Ask something new")

if question:
      vectordb = st.session_state.get("vectordb",None)
      if not vectordb:
            with st.message("assistent"):
                  st.write("you need to provide PDF")
                  st.stop()


search_results = vectordb.similarity_search(question,k = 3)
pdf_extract = "/n".join([result.page_content for result in search_results])

prompt[0] = {
      "role":"system",
      "content":prompt_template.format(pdf_extract = pdf_extract),
}


prompt.append({"role":"user","content":"question"})
with st.chat_message("assistant"):
      st.write(question)

with st.chat_message("assistant"):
      botmsg = st.empty()

response = []
result = ''
# Initialize an empty response list and result string
response = []
cohere_client = cohere.Client(cohere.api_key)
# Generate the response using Cohere's generate function
cohere_response = cohere_client.generate(
    model='xlarge',
    prompt=prompt_template.format(pdf_extract=pdf_extract),  # Format prompt with the PDF extract
    max_tokens=200,  # Maximum tokens to generate
    temperature=0.5  # Controls the randomness of the response
)

# Get the generated text from the Cohere response
result = cohere_response.generations[0].text.strip()

# Add the result to the response list
response.append(result)

# Write the response from the assistant
botmsg.write(result)

# Add the assistant's response to the prompt list
prompt.append({"role": "assistant", "content": result})

# Store the updated prompt in the session state
st.session_state["prompt"] = prompt
