import os
os.environ["OPENAI_API_KEY"] = "sk-proj-xjq9DOwj8GYSslVRUK0nT3BlbkFJ0CUm3yHRrYONTNYV3fQB"

# For retriving the data from the website
import bs4
from bs4 import BeautifulSoup as Soup 
from langchain.document_loaders.recursive_url_loader import RecursiveUrlLoader 
# For accessing the database
from langchain.vectorstores.pgvector import PGVector 
# For Embedding and accessing LLM Model
from langchain.embeddings import OpenAIEmbeddings 
from langchain.text_splitter import RecursiveCharacterTextSplitter 
from langchain.chat_models import ChatOpenAI 
# For Rag with LangChain
from langchain.prompts import ChatPromptTemplate
from langchain.memory import ConversationSummaryMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import OpenAIEmbeddings


# For Chat Interface
import gradio as gr


url = "https://www.hexacluster.ai/"
loader = RecursiveUrlLoader(
    url=url, max_depth=3, extractor=lambda x: Soup(x, "html.parser").text
)
docs = loader.load()

from langchain.embeddings import OpenAIEmbeddings
embeddings = OpenAIEmbeddings()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
doc_chunks = text_splitter.split_documents(docs)

CONNECTION_STRING = "postgresql+psycopg2://postgres@localhost:5432/ml"
COLLECTION_NAME = "hexacluster_website"

vecdb = PGVector.from_documents(
    embedding=embeddings,
    documents=doc_chunks,
    collection_name=COLLECTION_NAME,
    connection_string=CONNECTION_STRING,
    pre_delete_collection=True,
)

retriever = vecdb.as_retriever(search_kwargs={"k": 2})

template = """You are representing HexaCluster. Consider yourself as an intelligent Sales person from HexaCluster. Answer to be the below question in a pleasing tone from the information available in the below context.
{context}
Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

memory = ConversationSummaryMemory(
    llm=llm, memory_key="chat_history", return_messages=True
)

qa = ConversationalRetrievalChain.from_llm(
    llm, 
    retriever=retriever, 
    memory=memory, 
    combine_docs_chain_kwargs={"prompt": prompt}
)

def qa_answer(message, history):
    return qa(message)["answer"]

demo = gr.ChatInterface(qa_answer)

if __name__ == "__main__":
    demo.launch()