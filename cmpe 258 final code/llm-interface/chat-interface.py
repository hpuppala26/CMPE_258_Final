import gradio as gr
from dotenv import load_dotenv
from langchain_community.llms import Ollama
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings
import argparse
import time

count = -1
parser = argparse.ArgumentParser()
parser.add_argument("--port", type=int, default=7860)
parser.add_argument("--public", action="store_true", default=False)
args = parser.parse_args()

load_dotenv()


vector = FAISS.load_local(
    "final_extract_from_json_pdf_depth2",
    OpenAIEmbeddings(model="text-embedding-3-small"),
    allow_dangerous_deserialization=True,
)


class MaternaCare:
    def __init__(self, vector_store, top_k=3):
        self.main_llm_model = Ollama(model="llama3:instruct")
        self.retrieval_evaluator = Ollama(model="phi3:instruct")
        self.vector_store = vector_store
        self.top_k = top_k

        self.system_prompt = ChatPromptTemplate.from_template(
            """
            You are MaternaCare, a helpful Pregnancy Women assistant chatbot. Answer the question warmly, specific and very concise to the question asked, but you must answer it completely
            Check if the following context has information related to Question: {query}
            Context: {context}
            Content relevant?: {context_relevant} 
            If the content is not relevant, simply rationally say that you don't have the information.

            If at places, the information is a lot, put them in points. If you have good information to answer it, if you don't think its right, deny to answer
            Definitely output any links if you have them and they were helpful in answering the question
            """
        )

        self.evaluator_prompt = ChatPromptTemplate.from_template(
            """
            Your job is to evaluate the if the context is relevant to a specific question asked such that the context can answer the question in a helpful and relatable manner.
            Context: {context}
            Question: {query}
            if not relevant, output one word, "no", else output "yes". Just one word output
            """
        )

        self.evaluator_chain = self.evaluator_prompt.pipe(self.retrieval_evaluator)
        self.system_chain = self.system_prompt.pipe(self.main_llm_model)

    def answer_query(self, query):
        self.query = query
        self.context_documents = (
            self.vector_store.similarity_search_with_relevance_scores(
                query, k=self.top_k
            )
        )
        self.response = self.system_chain.invoke(
            {
                "query": self.query,
                "context": self.context_documents,
                "context_relevant": self.evaluator_chain.invoke(
                    {"context": self.context_documents, "query": self.query}
                ),
            }
        )
        print(self.context_documents)
        return self.response


s = MaternaCare(vector, top_k=5)


def respond(message, history):
    response = s.answer_query(message)
    # print(response, type(response))
    if isinstance(response, dict):
        return response["answer"]
    return response


gr.ChatInterface(fn=respond).launch(share=args.public)
