import asyncio
import json
import logging
import os
from datetime import datetime, timezone
from logging import INFO
from typing import List, Any

from dotenv import load_dotenv

from langchain_core.documents import Document

from graphiti_core import Graphiti
from graphiti_core.nodes import EpisodeType
from graphiti_core.search.search_config_recipes import NODE_HYBRID_SEARCH_RRF
from graphiti_core.llm_client.gemini_client import GeminiClient, LLMConfig
from graphiti_core.embedder.gemini import GeminiEmbedder, GeminiEmbedderConfig
from graphiti_core.cross_encoder.gemini_reranker_client import GeminiRerankerClient


#################################################
# CONFIGURATION
#################################################
# Set up logging and environment variables for
# connecting to Neo4j database
#################################################

# Configure logging
logging.basicConfig(
    level=INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)

# Neo4j connection parameters
# Make sure Neo4j Desktop is running with a local DBMS started
neo4j_uri = os.environ.get('NEO4J_URI', 'bolt://localhost:7687')
neo4j_user = os.environ.get('NEO4J_USER', 'neo4j')
neo4j_password = os.environ.get('NEO4J_PASSWORD', 'password')

if not neo4j_uri or not neo4j_user or not neo4j_password:
    raise ValueError('NEO4J_URI, NEO4J_USER, and NEO4J_PASSWORD must be set')

#################################################
# INITIALIZATION
#################################################
# Connect to Neo4j and set up Graphiti indices
# This is required before using other Graphiti
# functionality
#################################################

# Initialize Graphiti with Neo4j connection

# Google API key configuration
api_key = os.getenv('GOOGLE_API_KEY')

# Initialize Graphiti with Gemini clients
graphiti = Graphiti(
    neo4j_uri, neo4j_user, neo4j_password,
    # Initialize Gemini LLM client, embedder, and reranker
    llm_client=GeminiClient(
        config=LLMConfig(
            api_key=api_key,
            model="gemini-2.5-flash-lite"
        )
    ),
    embedder=GeminiEmbedder(
        config=GeminiEmbedderConfig(
            api_key=api_key,
            embedding_model="embedding-001"
        )
    ),
    cross_encoder=GeminiRerankerClient(
        config=LLMConfig(
            api_key=api_key,
            model="gemini-2.5-flash-lite"
        )
    )
)

def read_json_files_from_directory_and_convert_to_documents(directory: str) -> List[Document]:
    raw_docs = {}
    # Load dummy documents
    for file in os.listdir(directory):
        if file.endswith(".json"):
            with open(os.path.join(directory, file)) as f:
                raw_docs[file] = json.load(f)
    docs = [Document(page_content=json.dumps(doc), metadata={"type": EpisodeType.json, "description": f"Document from {file_name}", "group_id": file_name}) for file_name, doc in raw_docs.items()]
    return docs

# Converter function: LangChain Document(s) -> Graphiti episode dicts
def convert_langchain_documents_to_episodes(documents: List[Document]) -> List[dict]:
    """
    Convert LangChain Document objects to Graphiti episode dicts.
    Each Document should have .page_content (str) and .metadata (dict).
    If metadata contains 'type', use it, else default to text.
    """
    episodes = []
    for doc in documents:
        # Determine episode type
        episode_type = doc.metadata.get('type', EpisodeType.text)
        # If the content is JSON-like, treat as JSON episode
        if episode_type == EpisodeType.json or isinstance(doc.page_content, dict):
            content = doc.page_content if isinstance(doc.page_content, dict) else json.loads(doc.page_content)
        else:
            content = doc.page_content
        episodes.append({
            'content': content,
            'type': episode_type,
            'description': doc.metadata.get('description', ''),
            'group_id': doc.metadata.get('group_id', None),
        })
    return episodes

# Initialize the graph database with graphiti's indices. This only needs to be done once.
async def build_and_populate_graph(documents: List[Document], with_communities: bool = True):
    # Build the graph structure in Neo4j using Graphiti
    await graphiti.build_indices_and_constraints()

    episodes = convert_langchain_documents_to_episodes(documents)

    # Add episodes to the graph
    for i, episode in enumerate(episodes):
        await graphiti.add_episode(
            name=f'Freakonomics Radio {i}',
            episode_body=episode['content'] if isinstance(episode['content'], str) else json.dumps(episode['content']),
            source=episode['type'],
            source_description=episode['description'],
            group_id=episode['group_id'],
            reference_time=datetime.now(timezone.utc),
        )
        print(f'Added episode: Freakonomics Radio {i} ({episode["type"].value})')

    if with_communities:
        await graphiti.build_communities()

    


#################################################
# BASIC SEARCH
#################################################
# The simplest way to retrieve relationships (edges)
# from Graphiti is using the search method, which
# performs a hybrid search combining semantic
# similarity and BM25 text retrieval.
#################################################

# Perform a hybrid search combining semantic similarity and BM25 retrieval
async def basic_search(query: str):
    print("\nSearching for: 'Who was the California Attorney General?'")
    results = await graphiti.search('Who was the California Attorney General?')

    # Print search results
    print('\nSearch Results:')
    for result in results:
        print(f'UUID: {result.uuid}')
        print(f'Fact: {result.fact}')
        if hasattr(result, 'valid_at') and result.valid_at:
            print(f'Valid from: {result.valid_at}')
        if hasattr(result, 'invalid_at') and result.invalid_at:
            print(f'Valid until: {result.invalid_at}')
        print('---')

    return results


#################################################
# CENTER NODE SEARCH
#################################################
# For more contextually relevant results, you can
# use a center node to rerank search results based
# on their graph distance to a specific node
#################################################

# Use the top search result's UUID as the center node for reranking
async def center_node_search(results: List[Any]): # TODO check type of results
    if results and len(results) > 0:
        # Get the source node UUID from the top result
        center_node_uuid = results[0].source_node_uuid

        print('\nReranking search results based on graph distance:')
        print(f'Using center node UUID: {center_node_uuid}')

        reranked_results = await graphiti.search(
            'Who was the California Attorney General?', center_node_uuid=center_node_uuid
        )

        # Print reranked search results
        print('\nReranked Search Results:')
        for result in reranked_results:
            print(f'UUID: {result.uuid}')
            print(f'Fact: {result.fact}')
            if hasattr(result, 'valid_at') and result.valid_at:
                print(f'Valid from: {result.valid_at}')
            if hasattr(result, 'invalid_at') and result.invalid_at:
                print(f'Valid until: {result.invalid_at}')
            print('---')
        
        return reranked_results
    else:
        print('No results found in the initial search to use as center node.')


#################################################
# NODE SEARCH USING SEARCH RECIPES
#################################################
# Graphiti provides predefined search recipes
# optimized for different search scenarios.
# Here we use NODE_HYBRID_SEARCH_RRF for retrieving
# nodes directly instead of edges.
#################################################

# Example: Perform a node search using _search method with standard recipes
async def node_search_using_search_recipes(query: str):
    print(
        '\nPerforming node search using _search method with standard recipe NODE_HYBRID_SEARCH_RRF:'
    )

    # Use a predefined search configuration recipe and modify its limit
    node_search_config = NODE_HYBRID_SEARCH_RRF.model_copy(deep=True)
    node_search_config.limit = 5  # Limit to 5 results

    # Execute the node search
    node_search_results = await graphiti._search(
        query=query,
        config=node_search_config,
    )

    # Print node search results
    print('\nNode Search Results:')
    for node in node_search_results.nodes:
        print(f'Node UUID: {node.uuid}')
        print(f'Node Name: {node.name}')
        node_summary = node.summary[:100] + '...' if len(node.summary) > 100 else node.summary
        print(f'Content Summary: {node_summary}')
        print(f'Node Labels: {", ".join(node.labels)}')
        print(f'Created At: {node.created_at}')
        if hasattr(node, 'attributes') and node.attributes:
            print('Attributes:')
            for key, value in node.attributes.items():
                print(f'  {key}: {value}')
        print('---')

    return node_search_results

async def main():
    try:
        docs = read_json_files_from_directory_and_convert_to_documents("Documents")

        # Build and populate the graph with the dummy document
        await build_and_populate_graph(docs, with_communities=False)

        # Helper function to process search results into a list (for center_node_search)
        async def process_search_results(search_results):
            # If search_results is an object with .results, extract them
            if hasattr(search_results, 'results'):
                return search_results.results
            # If it's already a list, return as is
            return search_results

        # Perform a search using the _search method
        search_results = await basic_search('Who was the California Attorney General?')

        print(f'\nTotal search results: {len(search_results)}\n\n\nSearch Results:')
        for result in search_results:
            print(f' - {result}\n\n')

        # Process the search results
        processed_results = await process_search_results(search_results)
        print(f'\nTotal processed results: {len(processed_results)}\n\n\nProcessed Results:')
        for result in processed_results:
            print(f' - {result}')

        # Perform a center node search
        # await center_node_search(processed_results)

        # Perform a node search using search recipes
        # await node_search_using_search_recipes('California Attorney General')

    finally:
        #################################################
        # CLEANUP
        #################################################
        # Always close the connection to Neo4j when
        # finished to properly release resources
        #################################################

        # Close the connection
        await graphiti.close()
        print('\nConnection closed')


if __name__ == '__main__':
    asyncio.run(main())