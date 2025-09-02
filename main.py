import os
import asyncio
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_core.language_models import BaseLanguageModel
from langchain_openai import ChatOpenAI, OpenAIEmbeddings 
from langchain_graph_retriever import GraphRetriever
from langchain_core.vectorstores import InMemoryVectorStore
from graph_retriever.strategies import Eager
from langchain_graph_retriever import GraphRetriever

load_dotenv()

if __name__ == "__main__":
    summaries = [
        Document(page_content="GM was founded in 1908 in Flint, Michigan, and quickly became the world's largest automaker.", metadata={"id": "1"}),
        Document(page_content="Under Alfred Sloan, GM pioneered the multi-brand strategy and annual model change.", metadata={"id": "2"}),
        Document(page_content="GM plans to exclusively sell electric vehicles by 2035, investing heavily in EV technology.", metadata={"id": "3"}),
        Document(page_content="Cruise, GM's subsidiary, develops self-driving technology in urban U.S. markets.", metadata={"id": "4"}),
    ]

    # ---------------------------
    # Initialize LLM + embeddings
    # ---------------------------
    llm: BaseLanguageModel = ChatOpenAI(model="gpt-5-nano")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    vector_store = InMemoryVectorStore.from_documents(
        documents=summaries,
        embedding=embeddings,
    )


    traversal_retriever = GraphRetriever(
        store = vector_store,
        # edges = [("habitat", "habitat"), ("origin", "origin")],
        strategy = Eager(k=5, start_k=1, max_depth=2),
    )

    results = traversal_retriever.invoke("How has GMs strategy evolved over time?")

    for doc in results:
        print(f"{doc.id}: {doc.page_content}")