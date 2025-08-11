#import necessary libraries

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate, load_prompt
from pydantic import BaseModel, Field
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

# Connecting the prompt template
from TZ_prompt import prompt_template

from dotenv import load_dotenv
import os

# Load environment variables -> this is necessary to access OpenAI API keys
load_dotenv()

# Load the PromptTemplate
template = load_prompt("TZ_prompt_template.json")

# Define the schema of the output -> Pydantic model
class TechZoneAnalysis(BaseModel):
    problem_summary: str = Field(description="A clear and concise summary of the problem.")
    root_cause_analysis: list[str] = Field(description="List of all possible root cause analyses.")
    solutions: list[str] = Field(description="List of all relevant solutions, each actionable and specific.")
    techzone_links: list[str] = Field(description="Links to the most relevant TechZones.")
    CDETS_numbers: list[str] = Field(description="CDETS numbers associated with the TechZones, if available.")

# Define the LLM model
llm_model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7, max_tokens=1024)

# RAG (Retrieval-Augmented Generation)
# Function to load documents, split them into chunks, create embeddings and store them in vector store
def create_vector_store():
    """
    Function to create a vector store from the TechZones document.
    This function will be called only once to create the vector store.
    Steps include:
    1. Load the TechZones document
    2. Split the document into chunks
    3. Create embeddings for the chunks and store them in a vector database
    """
    PERSIST_DIR = "chroma_db"

    # Initialize the embeddings model
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    if not os.path.exists(PERSIST_DIR):
        # If the directory does not exist, create it and initialize the vector store
        print("First run: Creating Chroma DB...")

        # Load the TechZone document
        loader = TextLoader("TechZone.txt")
        documents = loader.load()
        print("Number of documents : ",len(documents))

        # Split the document into chunks
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        chunks = splitter.split_documents(documents)
        print("Number of Chunks formed : ",len(chunks))
        
        # Store the chunks in a vector database
        vector_store = Chroma.from_documents(
            chunks,
            embedding=embeddings,
            persist_directory=PERSIST_DIR
        )
        print("Vector store created and persisted.")
    else:
        # Load the existing vector store
        print("Reloading existing Chroma DB...")
        vector_store = Chroma(
            persist_directory=PERSIST_DIR,
            embedding_function=embeddings
        )
    return vector_store

def analyse_query(TZ_link: str, TZ_description: str, problem_type: str, component: str) -> TechZoneAnalysis:
    
    """
    Function to analyze the query using the LLM and the vector store.
    Steps include:
    1. Create a retriever for relevant chunks based on the user's query
    2. Retrieve relevant chunks based on the user's input
    3. Generate a response using the retrieved chunks and the language model
    """
    print("Analyzing query...")

    # 1. Create or load the vector store
    print("Creating or loading the vector store...")
    vector_store = create_vector_store()
    print("Vector store created or loaded.")

    # 2. Ensure the vector store is not empty
    if not vector_store:
        raise ValueError("The vector store is empty. Please ensure the TechZones document is loaded correctly.")
    print("Vector store is not empty.")

    # 4. Create a retriever for relevant chunks based on the user's query
    retriever = vector_store.as_retriever(
                                         search_type="mmr",
                                         search_kwargs={"k": 5}
                                        )
    print("Retriever created with search type 'mmr' and k=5.")

    # 5. Retrieve relevant chunks based on the user's input -> Prepare the Context
    # If retriever.invoke() fails, we will use the fallback method i.e. retriever.get_relevant_documents()
    try:
        print("Retrieving relevant documents...")
        retrieved_docs = retriever.invoke(TZ_description)
        print(len(retrieved_docs), "relevant documents retrieved.")
    except Exception as e:
        print("Error occurred while retrieving documents:", e)
        retrieved_docs = retriever.get_relevant_documents(TZ_description)
        print(len(retrieved_docs), "relevant documents retrieved using fallback method.")

    context = "\n".join ([
        f"Description: {doc.page_content}, "
        f"Link: {doc.metadata.get('link','N/A')}, "
        f"CDETS: {doc.metadata.get('CDETS','N/A')}" 
        for doc in retrieved_docs
    ])

    # 6. Invoke the prompt template along with retrieved context
    formatted_prompt = template.invoke({
        "TZ_description" : TZ_description,
        "problem_type" : problem_type,
        "component" : component,
        "context" : context
    })

    # 7. Invoke the LLM with the formatted prompt and structured output
    structured_output = llm_model.with_structured_output(TechZoneAnalysis)
    response = structured_output.invoke(formatted_prompt)
    
    return response