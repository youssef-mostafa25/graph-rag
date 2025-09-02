import os
import asyncio
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.retrievers.graph_rag import GraphRAGIndexer, GraphRAGRetriever

google_api_key = os.getenv["GOOGLE_API_KEY"]
OUTPUT_DIR = "@/output"  # 

# Initialize embeddings with Gemini
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",   # Gemini embedding model
    google_api_key=google_api_key
)

# Initialize LLM with Gemini
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",         # Gemini chat/LLM model
    google_api_key=google_api_key
)

# Initialize the indexer
indexer = GraphRAGIndexer(
    llm=llm,
    embeddings=embeddings,
    storage_path=OUTPUT_DIR  # where the graph + vectors will be persisted
)

# Initialize the retriever
retriever = GraphRAGRetriever(storage_path=OUTPUT_DIR)

# Build the index
async def build_index(docs: list[Document]):
    try:
        indexer = GraphRAGIndexer(storage_path=OUTPUT_DIR)
        indexer.add_documents(docs)
        print("Index built successfully")
    except Exception as e:
        print(f"Error building index: {e}")
        raise e

# Query the index
def query(user_input: str):
    return retriever.invoke(user_input)

if __name__ == "__main__":
    # langchain docs
    summaries = [
        Document(page_content="GM was founded in 1908 in Flint, Michigan, and quickly expanded through acquisitions, becoming the world’s largest automaker by the mid-20th century.", metadata={"id": "1"}),
        Document(page_content="Under Alfred Sloan, GM pioneered the multi-brand strategy and annual model change, transforming the auto industry’s marketing approach.", metadata={"id": "2"}),
        Document(page_content="The company operates worldwide, with production facilities and sales networks in North America, South America, Europe, and Asia.", metadata={"id": "3"}),
        Document(page_content="Chevrolet, Cadillac, Buick, and GMC make up GM’s core brands, each targeting different customer segments.", metadata={"id": "4"}),
        Document(page_content="Chevrolet is GM’s mass-market brand, famous for models like the Silverado pickup and Corvette sports car.", metadata={"id": "5"}),
        Document(page_content="Cadillac represents GM’s luxury brand, competing with Mercedes-Benz, BMW, and Lexus in premium markets.", metadata={"id": "6"}),
        Document(page_content="GM filed for Chapter 11 bankruptcy in 2009 and received a U.S. government bailout, restructuring successfully into a leaner company.", metadata={"id": "7"}),
        Document(page_content="After restructuring, GM eliminated brands like Pontiac, Saturn, and Hummer, while focusing on global core brands.", metadata={"id": "8"}),
        Document(page_content="GM plans to exclusively sell electric vehicles by 2035, investing heavily in EV technology.", metadata={"id": "9"}),
        Document(page_content="Ultium is GM’s modular EV battery system, designed for scalability across vehicles from small cars to large trucks.", metadata={"id": "10"}),
        Document(page_content="The GMC Hummer EV, Chevrolet Silverado EV, and Cadillac Lyriq are leading models in GM’s electric lineup.", metadata={"id": "11"}),
        Document(page_content="Cruise, GM’s subsidiary, develops self-driving technology and has launched pilot programs in urban U.S. markets.", metadata={"id": "12"}),
        Document(page_content="GM is transitioning to software-first vehicles, with over-the-air updates and digital services as revenue streams.", metadata={"id": "13"}),
        Document(page_content="The company targets carbon neutrality by 2040 and aims to source all U.S. operations’ electricity from renewable energy by 2025.", metadata={"id": "14"}),
        Document(page_content="GM Defense develops military vehicles and technologies, including hydrogen fuel-cell powered trucks for defense applications.", metadata={"id": "15"}),
        Document(page_content="GM collaborates with LG Energy Solution on Ultium Cells battery plants, and with Honda on fuel cell and EV projects.", metadata={"id": "16"}),
        Document(page_content="GM competes against Ford, Toyota, Volkswagen, Tesla, and emerging EV startups in both traditional and electric markets.", metadata={"id": "17"}),
        Document(page_content="Despite shifting to EVs, GM continues to lead in profitable trucks and SUVs, including the GMC Sierra and Chevrolet Tahoe.", metadata={"id": "18"}),
        Document(page_content="GM exited some regions like Europe (selling Opel to PSA in 2017) but remains strong in North and South America and China.", metadata={"id": "19"}),
        Document(page_content="GM has reported steady profits driven by high-margin trucks and SUVs, while scaling EV and software investments for long-term growth.", metadata={"id": "20"}),
    ]

    asyncio.run(build_index(summaries))

    asyncio.run(query(user_input="What are the top themes in this story?"))