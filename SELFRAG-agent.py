from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain import hub
from langchain_core.output_parsers import StrOutputParser

from dotenv import load_dotenv
from typing import List
from typing_extensions import TypedDict

from langgraph.graph import END, StateGraph, START

from prompts import DOCUMENT_GRADER_PROMPT, HALLUCINATION_GRADER_PROMPT, ANSWER_GRADER_PROMPT

KNOWLEDGE_BASE_URLS = [
    "https://www.linkedin.com/pulse/parallel-execution-nodes-langgraph-enhancing-your-graph-prateek-qqwrc/",
    "https://www.linkedin.com/pulse/tool-calling-langchain-do-more-your-ai-agents-saurav-prateek-so20c",
]

# DEFINING TYPE OF ATTRIBUTES OF GRAPHSTATE
class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        documents: list of documents
        model: LLM model used for generation
        vector_store: vector store for RAG
        hallucinated: whether the generation is grounded in documents
        valid_answer: whether the generation answers the question
    """

    question: str
    generation: str
    documents: List[str]
    model: ChatOpenAI
    vector_store: Chroma
    hallucinated: bool
    valid_answer: bool

    