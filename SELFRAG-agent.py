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

# DATAMODEL FOR GRADING DOCUMENTS
class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""

    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )

# DATAMODEL FOR GRADING HALLUCINATIONS
class GradeHallucinations(BaseModel):
    """Binary score for hallucination present in generation answer."""

    binary_score: str = Field(
        description="Answer is grounded in the facts, 'yes' or 'no'"
    )

# DATAMODEL FOR GRADING THE FINAL ANSWER
class GradeAnswer(BaseModel):
    """Binary score to assess answer addresses question."""

    binary_score: str = Field(
        description="Answer addresses the question, 'yes' or 'no'"
    )   

def create_model(state):
    print("---CREATE GPT MODEL---")
    state['model'] = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    return state

# FUNCTION TO BUILD VECTOR STORE
def build_vector_store(state):
    print("---BUILD VECTOR STORE---")
    docs = [WebBaseLoader(url).load() for url in KNOWLEDGE_BASE_URLS]
    docs_list = [item for sublist in docs for item in sublist]

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=250, chunk_overlap=0
    )
    doc_splits = text_splitter.split_documents(docs_list)

    # Add to vectorDB
    vector_store = Chroma.from_documents(
        documents=doc_splits,
        collection_name="rag-chroma",
        embedding=OpenAIEmbeddings(),
    )
    state['vector_store'] = vector_store.as_retriever()

    return state