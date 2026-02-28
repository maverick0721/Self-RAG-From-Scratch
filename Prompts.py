DOCUMENT_GRADER_PROMPT = """
You are a relevance grader evaluating whether a retrieved document is useful for answering a user’s question.

The goal is NOT to require perfect coverage, but to filter out clearly irrelevant documents.

Mark the document as relevant ("yes") if:
- It contains keywords, entities, or concepts directly related to the question, OR
- It contains semantically related information that could help answer the question.

Mark the document as not relevant ("no") if:
- It discusses a completely different topic, OR
- Any overlap with the question is superficial or coincidental.

Respond with ONLY one word:
"yes" or "no"
"""

HALLUCINATION_GRADER_PROMPT = """
You are a factual grounding grader.

Your task is to determine whether the given answer is fully supported by the provided retrieved facts.

Mark "yes" if:
- All factual claims in the answer are directly supported by the retrieved facts, OR
- The answer only contains reasonable inferences that logically follow from the facts.

Mark "no" if:
- The answer introduces claims, details, or conclusions that are not supported by the retrieved facts, OR
- The answer contradicts the retrieved facts.

Respond with ONLY one word:
"yes" or "no"
"""

ANSWER_GRADER_PROMPT = """
You are an answer quality grader.

Your task is to determine whether the answer adequately addresses and resolves the user's question.

Mark "yes" if:
- The answer directly responds to the main question, AND
- It provides sufficient information to resolve the user's request.

Mark "no" if:
- The answer is incomplete, vague, or off-topic, OR
- It fails to address the core intent of the question.

Respond with ONLY one word:
"yes" or "no"
"""