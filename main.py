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

async def basic_search(user_query: str):
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

async def local_search(user_query: str):
    # Load required DataFrames from index output
    entities_df = pd.read_parquet(OUTPUT_DIR / "entities.parquet")
    communities_df = pd.read_parquet(OUTPUT_DIR / "communities.parquet")
    community_reports_df = pd.read_parquet(OUTPUT_DIR / "community_reports.parquet")
    text_units_df = pd.read_parquet(OUTPUT_DIR / "text_units.parquet")
    relationships_df = pd.read_parquet(OUTPUT_DIR / "relationships.parquet")
    covariates_path = OUTPUT_DIR / "covariates.parquet"
    covariates_df = pd.read_parquet(covariates_path) if covariates_path.exists() else None

    context, response = await api.local_search(
        config=graphrag_config,
        entities=entities_df,
        communities=communities_df,
        community_reports=community_reports_df,
        text_units=text_units_df,
        relationships=relationships_df,
        covariates=covariates_df,
        community_level=1,  # You may want to make this configurable
        response_type="text",  # You may want to make this configurable
        query=user_query,
    )
    print("Response:", context)

async def global_search(user_query: str):
    # Load required DataFrames from index output
    entities_df = pd.read_parquet(OUTPUT_DIR / "entities.parquet")
    communities_df = pd.read_parquet(OUTPUT_DIR / "communities.parquet")
    community_reports_df = pd.read_parquet(OUTPUT_DIR / "community_reports.parquet")

    context, response = await api.global_search(
        config=graphrag_config,
        entities=entities_df,
        communities=communities_df,
        community_reports=community_reports_df,
        community_level=1,  # You may want to make this configurable
        dynamic_community_selection=False,  # You may want to make this configurable
        response_type="text",  # You may want to make this configurable
        query=user_query,
    )
    print("Response:", context)

async def drift_search(user_query: str): # token limit reached
    # Load required DataFrames from index output
    entities_df = pd.read_parquet(OUTPUT_DIR / "entities.parquet")
    communities_df = pd.read_parquet(OUTPUT_DIR / "communities.parquet")
    community_reports_df = pd.read_parquet(OUTPUT_DIR / "community_reports.parquet")
    text_units_df = pd.read_parquet(OUTPUT_DIR / "text_units.parquet")
    relationships_df = pd.read_parquet(OUTPUT_DIR / "relationships.parquet")

    context, response = await api.drift_search(
        config=graphrag_config,
        entities=entities_df,
        communities=communities_df,
        community_reports=community_reports_df,
        text_units=text_units_df,
        relationships=relationships_df,
        community_level=1,  # You may want to make this configurable
        response_type="text",  # You may want to make this configurable
        query=user_query,
    )
    print("Response:", context)

if __name__ == "__main__":
    # asyncio.run(build_index_from_files())
    asyncio.run(basic_search("What are the top themes in this story?"))
    asyncio.run(local_search("What are the top themes in this story?"))
    asyncio.run(global_search("What are the top themes in this story?"))
    # asyncio.run(drift_search("What are the top themes in this story?"))

