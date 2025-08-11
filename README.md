# TechZone Analyser
A problem analysis and summarization tool for TechZone issues, powered by Retrieval-Augmented Generation (RAG).

## Problem Statement
TechZones are internal templates where users report issues encountered in various system components, including problem descriptions, issue types, and affected components. However, manually analyzing these entries to determine the root cause, discover previous similar issues, and find corresponding already solved bug-ids is time-consuming and inconsistent.

This project solves that by allowing users to input a TechZone link, problem description, issue type, and component. The **TechZone Analyzer** then:

- Summarizes the problem
- Lists possible solutions and root causes
- Links to relevant past TechZone reports
- Maps associated CDETS IDs
  
## Dataset Description
The dataset used in this project consists of structured records capturing real-world issues reported in the TechZone system. Each entry in the dataset includes several key fields that describe the technical problem and its context, enabling efficient retrieval and analysis.

Each record includes:

TZ_link: A unique URL pointing to the specific TechZone issue.

TZ_description: A detailed textual description of the problem as reported by the user.

Problem_type: A category label (e.g., Crash, Performance, Security, etc.) indicating the type of issue.

Component: The specific software or hardware component affected by the issue.

Root_Causes: A list of possible or identified root causes for the issue.

Solutions: Suggested or implemented solutions to resolve the problem.

CDETS: One or more associated bug identifiers linked to the issue.

## Libraries used
| Library               | Description                                                |
| --------------------- | ---------------------------------------------------------- |
| `langchain`           | Framework for developing LLM-powered applications          |
| `langchain-core`      | Provides core building blocks for LangChain apps           |
| `langchain-community` | Community-contributed integrations and tools for LangChain |
| `langchain-openai`    | Integration module for using OpenAI with LangChain         |
| `openai`              | Direct interface to OpenAI's API for LLMs                  |
| `pydantic`            | Ensures type validation and structured data handling       |
| `langchain-chroma`    | Integration between LangChain and Chroma vector DB         |
| `chromadb`            | Vector database for storing and querying embedded data     |
| `streamlit`           | Used to create the web interface for user interaction      |
| `python-dotenv`       | Loads environment variables from a `.env` file             |
| `pandas`              | Data manipulation and analysis tool                        |
| `numpy`               | Numerical operations and array handling                    |
| `scikit-learn`        | ML utilities for preprocessing and evaluation              |
| `matplotlib`          | Visualization and charting for analysis                    |


## Methodology used
The core methodology used is Retrieval-Augmented Generation (RAG):

Embedding and Storage: The dataset entries are vectorized using embeddings and stored in a vector database.

Querying: User inputs are embedded and matched against the stored dataset to retrieve the most relevant context.

Generation: An LLM (via OpenAI API) is prompted using the retrieved context to generate a summarized response including:
- Summary of the problem
- Similar TZ issues
- Root cause analysis
- Solutions and CDETS references

Web App: Streamlit is used to build a user-friendly interface.

## Results
The final application enables users to input issue details and instantly receive:
- A summary of the issue
- Similar past TechZone entries
- Possible root causes and actionable solutions
- Linked CDETS IDs for reference

This greatly reduces time spent on root cause analysis and avoids repetitive issue handling.

## Future work
- Every new TechZone queried should be added to the Database.
- Add issue tagging and severity prediction using ML
- Expand to support unstructured or semi-structured TechZone formats
