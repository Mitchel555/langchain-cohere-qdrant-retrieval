from flask import Flask, request
from flask_cors import CORS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import ElasticVectorSearch, Pinecone, Weaviate, FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
import json

# Loading environment variables
import os
from dotenv import load_dotenv
load_dotenv()
openai_api_key = os.environ.get('openai_api_key')
cohere_api_key = os.environ.get('cohere_api_key')
qdrant_url = os.environ.get('qdrant_url')
qdrant_api_key = os.environ.get('qdrant_api_key')

#Flask config
app = Flask(__name__)
CORS(app)

# Test default route
@app.route('/')
def hello_world():
    return {"Hello":"World"}

## Embedding code
from langchain.embeddings import CohereEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import Qdrant

@app.route('/embed', methods=['POST'])
def embed_pdf():
    collection_name = request.json.get("collection_name")
    file_url = request.json.get("file_url")

    loader = PyPDFLoader(file_url)
    docs = loader.load_and_split()

    embeddings = OpenAIEmbeddings(openai_api_key)  # replacing CohereEmbeddings with OpenAIEmbeddings

    qdrant = Qdrant.from_documents(docs, embeddings, url=qdrant_url, collection_name=collection_name, prefer_grpc=True, api_key=qdrant_api_key)
    
    return {"collection_name": qdrant.collection_name}

# Retrieve information from a collection
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from qdrant_client import QdrantClient
from langchain.prompts import PromptTemplate
from langchain.prompts.chat import ChatPromptTemplate, HumanMessagePromptTemplate

from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory

@app.route('/retrieve', methods=['POST'])
def retrieve_info():
    collection_name = request.json.get("collection_name")
    query = request.json.get("query")

    client = QdrantClient(url=qdrant_url, prefer_grpc=True, api_key=qdrant_api_key)

    embeddings = OpenAIEmbeddings(openai_api_key) 
    qdrant = Qdrant(client=client, collection_name=collection_name, embedding_function=embeddings.embed_query)
    search_results = qdrant.similarity_search(query, k=5)

    # Define the LLM and the prompt
    chat = ChatOpenAI(temperature=0)
    human_message_prompt = HumanMessagePromptTemplate(
        prompt=PromptTemplate(
            template="Zpracuj odpoved na zaklade uzivatelskeho vstupu: {query} Poskytnuty Text:{search_results} Pokud odpověď v tomto textu nenajdes, napiš že nevíš",
            input_variables=["query","search_results"],
        )
    )
    chat_prompt_template = ChatPromptTemplate.from_messages([human_message_prompt])

    # Define the chain
    chain = LLMChain(llm=chat, prompt=chat_prompt_template)

    # Prepare the input variables for the chain
    input_vars = {"search_results": search_results, "query": query}

    # Print the prompt
    # prompt = "Zpracuj odpoved na zaklade uzivatelskeho vstupu: {query} Poskytnuty Text:{search_results} Pokud odpověď v tomto textu nenajdes, napiš že nevíš".format(query=query, search_results=search_results)
    # print(f'Prompt sent to OpenAI: {prompt}')

    # Run the chain
    results = chain.run(input_vars)

    return {"results": results}
