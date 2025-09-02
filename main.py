import asyncio
from pathlib import Path
import pandas as pd
from graphrag.config.load_config import load_config
import graphrag.api as api


PROJECT_DIR = "./ragtest"
OUTPUT_DIR = Path(PROJECT_DIR) / "output"
graphrag_config = load_config(Path(PROJECT_DIR))


# 1) Build the GraphRAG index from files in ragtest/input
async def build_index_from_files() -> None:
    index_results = await api.build_index(config=graphrag_config)
    for wf in index_results:
        print(f"Workflow: {wf.workflow}, Status: {'Error' if wf.errors else 'Success'}")


# 2) Retrieve from the graph
async def retrieve(user_query: str):
    # Ensure index exists; build if missing
    output_dir = OUTPUT_DIR
    if not output_dir.exists() or not any(output_dir.iterdir()):
        await build_index_from_files()

    # Load required DataFrame(s) from index output
    text_units_path = OUTPUT_DIR / "text_units.parquet"
    text_units_df = pd.read_parquet(text_units_path)
    # Use a concrete search entrypoint (query is a module, not a callable)
    context, response = await api.basic_search(
        config=graphrag_config,
        text_units=text_units_df,
        query=user_query,
    )
    print("Response:", context)
    return context, response


if __name__ == "__main__":
    # asyncio.run(build_index_from_files())
    asyncio.run(retrieve("What are the top themes in this story?"))

