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

# FUNCTION TO RETRIEVE RELEVANT DOCUMENTS
def get_relevant_documents(state):
    """
    Retrieve documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    print("---RETRIEVE RELEVANT DOCUMENTS---")
    question = state["question"]

    # Retrieval
    vector_store = state["vector_store"]
    documents = vector_store.get_relevant_documents(question)
    state["documents"] = documents

    return state

# FUNCTION TO GRADE DOCUMENTS
def grade_documents(state):
    """
    Determines whether the retrieved documents are relevant to the question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates documents key with only filtered relevant documents
    """

    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state["documents"]

    structured_llm_grader = state["model"].with_structured_output(GradeDocuments)
    grade_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", DOCUMENT_GRADER_PROMPT),
            ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
        ]
    )
    retrieval_grader = grade_prompt | structured_llm_grader

    # SCORE EACH DOC
    filtered_docs = []
    for d in documents:
        score = retrieval_grader.invoke(
            {"question": question, "document": d.page_content}
        )
        grade = score.binary_score
        if grade == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(d)
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
            continue
    
    state['documents'] = filtered_docs
    return state


# FUNCTION TO DECIDE WHETHER TO GENERATE OR NOT
def decide_to_generate(state):
    """
    Determines whether to generate an answer, or re-generate a question.

    Args:
        state (dict): The current graph state

    Returns:
        str: Binary decision for next node to call
    """

    print("---ASSESS GRADED DOCUMENTS---")
    filtered_documents = state["documents"]

    if not filtered_documents:
        # All documents have been filtered check_relevance
        # We will re-generate a new query
        print(
            "---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, END ---"
        )
        return "end"
    else:
        # We have relevant documents, so generate answer
        print("---DECISION: GENERATE---")
        return "continue"