import os
import faiss
import requests
from io import BytesIO
import pdfplumber
from flask import Flask, request, jsonify
from flask_cors import CORS
from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import ElasticVectorSearch, Pinecone, Weaviate, FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.prompts.chat import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.prompts import MessagesPlaceholder, HumanMessagePromptTemplate, ChatPromptTemplate
from langchain.schema import HumanMessage
from langchain.prompts.chat import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate


os.environ["OPENAI_API_KEY"] = "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"

app = Flask(__name__)

@app.route('/embed', methods=['POST'])
def embed_text():
    index_name = request.json.get("index_name")
    file_url = request.json.get("file_url")
    
    # Download and extract text from PDF
    texts = download_and_extract_text(file_url)

    # Generate embeddings and store in FAISS index
    embeddings = OpenAIEmbeddings()
    store = FAISS.from_texts(texts, embeddings)

    # Write FAISS index to a file and clear the indexS
    # faiss.write_index(store.index, index_name)
    store.save_local(index_name)
    store.index = None

    return {"message": "Embedding completed and FAISS index saved successfully."}, 200

@app.route('/retrieve', methods=['POST'])
def retrieve_text():
    index_name = request.json.get("index_name")
    query = request.json.get("query")
    embeddings = OpenAIEmbeddings()
    index = FAISS.load_local(index_name, embeddings)
    # chain = load_qa_chain(ChatOpenAI(), chain_type="stuff")

    docs = index.similarity_search(query, k=3)

    chat = ChatOpenAI(temperature=0)
    human_message_prompt = HumanMessagePromptTemplate(
        prompt=PromptTemplate(
            template="{query}",
            input_variables=["query"],
        )
    )
    system_template = "Jsi chatbot Chetty, ktery odpovida pouze na zaklade techto svych vedomosti, odpovidej strucne a mluv jako kdyby jsi zastupoval Multimu: {docs}"
    system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)
    
    chat_prompt_template = ChatPromptTemplate.from_messages([human_message_prompt, system_message_prompt])

    # Define the chain
    chain = LLMChain(llm=chat, prompt=chat_prompt_template)
    input_vars = {"docs": docs, "query": query}
    answer = chain.run(input_vars)
    print(answer)
    # return jsonify({"answer": answer})
    return jsonify({"results": answer}), 200


def download_and_extract_text(file_url):
    # Send a HTTP request to the URL
    with requests.get(file_url) as r:
        r.raise_for_status()
        # Store the file in memory
        with BytesIO(r.content) as f:
            # Open the PDF file
            with pdfplumber.open(f) as pdf:
                raw_text = ''
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        raw_text += text

                text_splitter = CharacterTextSplitter(
                    separator="\n",
                    chunk_size=1000,
                    chunk_overlap=200,
                    length_function=len,
                )
                texts = text_splitter.split_text(raw_text)
    return texts

if __name__ == '__main__':
    app.run(debug=True)
