from pathlib import Path
import asyncio
from graphrag.config.load_config import load_config
from graphrag.index import InMemoryIndex
from graphrag.query import QueryEngine
from langchain_core.documents import Document

PROJECT_DIR = "@/ragtest"  # Your project root
graphrag_config = load_config(Path(PROJECT_DIR))

# Refactor the langchain docs to graphrag jsonl
def refactor_langchain_docs_to_graphrag_jsonl(docs: list[Document]) -> list[Document]:
    refactored_docs = [
    {"id": str(i), "text": doc.page_content, "metadata": doc.metadata}
    for i, doc in enumerate(docs)
    ]
    return refactored_docs

# Build the index
async def build_index(docs: list[Document]):
    try:
        index = InMemoryIndex()
        index.add_documents(docs)
        index.save(graphrag_config.project.paths.output)
    except Exception as e:
        print(f"Error building index: {e}")
        raise e

# Load the index
async def load_index() -> InMemoryIndex:
    index = InMemoryIndex()
    index.load(graphrag_config.project.paths.output)
    return index

# Get the engine
async def get_engine() -> QueryEngine:
    index = await load_index()
    engine = QueryEngine(index, graphrag_config)
    return engine

# Query the index
def query(user_input: str, engine: QueryEngine):
    return engine.query(user_input)

if __name__ == "__main__":
    # langchain docs
    summaries = [
        Document(page_content="Hello, world!", metadata={"id": "1"}),
        Document(page_content="Hello, world!", metadata={"id": "2"}),
        Document(page_content="Hello, world!", metadata={"id": "3"}),
    ]

    refactored_summaries = refactor_langchain_docs_to_graphrag_jsonl(summaries)
    asyncio.run(build_index(refactored_summaries))

    engine = asyncio.run(get_engine())
    asyncio.run(query(user_input="What are the top themes in this story?", engine=engine))