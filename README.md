# 🧠 Self-RAG from Scratch  
### A Self-Reflective Retrieval-Augmented Generation System with LangGraph

> 🧠 A self-reflective Retrieval-Augmented Generation (Self-RAG) system integrating document relevance grading, hallucination detection, and answer validation.  
> 🔍 Implements structured LLM-based evaluators to enforce factual grounding and response quality.  
> 📈 Demonstrates production-oriented RAG orchestration using conditional execution graphs and automated quality control.

---

## 1. Overview

This project implements a structured, self-correcting Retrieval-Augmented Generation (RAG) pipeline using:

- **LangGraph** (graph orchestration)
- **Chroma** (vector store)
- **OpenAI Embeddings**
- **Structured LLM grading with Pydantic**

Unlike naive RAG systems that simply retrieve and generate, this pipeline actively evaluates:

- Document relevance  
- Factual grounding (hallucination detection)  
- Answer quality  

The result is a more reliable and production-aligned RAG architecture.

---

## 2. Motivation

Standard RAG systems suffer from:

- Retrieval noise  
- Hallucinated outputs  
- Irrelevant or incomplete answers  
- No structured quality control  

This project introduces a **Self-RAG** workflow where the model:

1. Retrieves documents  
2. Grades document relevance  
3. Generates an answer  
4. Checks hallucinations  
5. Validates answer quality  

This transforms RAG into a **self-evaluating reasoning system**.

---

## 3. System Architecture

### High-Level Execution Flow

```
START
  ↓
Create Model
  ↓
Build Vector Store
  ↓
Retrieve Documents
  ↓
Grade Documents
  ↓ (if relevant)
Generate Answer
  ↓
Hallucination Check
  ↓
Answer Quality Check
  ↓
END
```

The pipeline is implemented using **LangGraph’s `StateGraph`**, enabling:

- Deterministic orchestration  
- Conditional branching  
- Structured state transitions  

---

## 4. Core Components

### 4.1 Vector Store Construction

- Source: External knowledge base URLs  
- Loader: `WebBaseLoader`  
- Splitter: `RecursiveCharacterTextSplitter`  
- Embeddings: `OpenAIEmbeddings`  
- Vector Database: `Chroma`  

Documents are chunked and embedded before being indexed.

---

### 4.2 Structured LLM Graders (LLM-as-Judge)

Three structured evaluators are implemented:

#### 1️⃣ Document Relevance Grader  
Determines if retrieved documents are useful for answering the question.

#### 2️⃣ Hallucination Grader  
Checks whether generated answers are fully grounded in retrieved documents.

#### 3️⃣ Answer Quality Grader  
Ensures the answer directly addresses the user’s query.

Each grader returns strictly:

```
"yes" or "no"
```

This enables deterministic conditional routing.

---

## 5. Conditional Execution Example

```python
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "continue": "generate_answer",
        "end": END,
    },
)
```

If no relevant documents are found, generation is skipped.

---

## 6. Technical Stack

| Component | Tool |
|------------|------|
| LLM | GPT-4o-mini |
| Orchestration | LangGraph |
| Vector Store | Chroma |
| Embeddings | OpenAIEmbeddings |
| Document Loader | WebBaseLoader |
| Chunking | RecursiveCharacterTextSplitter |
| Structured Output | Pydantic |

---

## 7. Installation

```bash
pip install -r requirements.txt
```

### requirements.txt

```
langchain
langchain-core
langchain-community
langchain-openai
langgraph
chromadb
python-dotenv
tiktoken
pydantic
langchain-text-splitters
beautifulsoup4
lxml
```

---

## 8. Usage

Unified one-command launcher:

```bash
chmod +x run.sh && ./run.sh --local
```

Docker run (build + run full demo):

```bash
./run.sh --docker
```

Docker Compose run:

```bash
./run.sh --compose
```

Manual Docker build and run:

```bash
docker build --ignorefile docker/.dockerignore -f docker/Dockerfile -t self-rag-from-scratch:latest .
docker run --rm --env-file .env self-rag-from-scratch:latest
```

Direct run:

```bash
python SELFRAG-agent.py
```

Optional programmatic usage:

```python
from importlib.machinery import SourceFileLoader

selfrag = SourceFileLoader("selfrag", "SELFRAG-agent.py").load_module()
response = selfrag.run_self_rag("What is RAG & how does it work?")
print(response.get("generation", "No generation returned"))
```

Smoke tests:

```bash
python -m unittest -q
```

The system will:

- Retrieve documents  
- Filter irrelevant content  
- Generate grounded answer  
- Check hallucination  
- Validate answer quality  

---

## 9. Key Contributions

- Designed a self-reflective RAG architecture  
- Implemented structured LLM grading mechanisms  
- Built conditional graph-based orchestration  
- Added hallucination detection before final output  
- Enforced answer relevance validation  
- Demonstrated production-style AI workflow design  

---

## 10. Production Implications

This architecture improves:

- Reliability of RAG systems  
- Grounding consistency  
- Explainability of pipeline steps  
- Control over hallucination risk  

It represents a move from naive retrieval systems toward **self-regulating LLM workflows**.

---

## 11. Conclusion

This project demonstrates how Retrieval-Augmented Generation systems can be enhanced through structured evaluation and graph-based orchestration.

By integrating document filtering, hallucination detection, and answer validation, the pipeline improves robustness and aligns more closely with production deployment standards.

---

## 12. Future Extensions

- Multi-hop retrieval  
- Iterative re-query loops  
- Confidence scoring  
- Hybrid local + remote embeddings  
- Tool-calling integration  
- Multi-agent RAG workflows
