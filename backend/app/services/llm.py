import json
import re

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate

from app.core.config import get_settings


settings = get_settings()


# ============================================================
# GROQ MODEL
# ============================================================

def _model():
    return ChatGroq(
        api_key=settings.groq_api_key,
        model=settings.groq_model,
        temperature=0,
    )


# ============================================================
# QUERY TRANSFORMATION
# ============================================================

QUERY_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a query transformation component for a document RAG system.

Your task is to rewrite the latest user question into ONE
self-contained retrieval query.

Rules:
1. Correct obvious spelling and grammar mistakes.
2. Resolve references such as "it", "that", "he", "this",
   using the conversation history when possible.
3. Preserve the user's original meaning.
4. Do not answer the question.
5. Do not add information that is not present in the conversation.
6. Return ONLY the rewritten retrieval query.
"""
    ),
    (
        "human",
        """Conversation history:

{history}

Latest question:

{question}
"""
    ),
])


async def transform_query(
    question: str,
    history: list[dict],
) -> str:

    history_text = "\n".join(
        f"{message['role']}: {message['content']}"
        for message in history[-8:]
    )

    if not history_text:
        history_text = "(no previous conversation)"

    result = await (
        QUERY_PROMPT | _model()
    ).ainvoke(
        {
            "history": history_text,
            "question": question,
        }
    )

    text = (
        result.content
        if isinstance(result.content, str)
        else str(result.content)
    )

    return text.strip() or question


# ============================================================
# JAYS SYSTEM PROMPT
# ============================================================

JAYS_SYSTEM = """You are Jays, a precise AI document assistant.

Your job is to answer questions using ONLY the supplied
retrieved document evidence.

Rules:

1. Never invent facts.

2. Never invent citations, page numbers, names, dates,
   statistics, or explanations.

3. If the retrieved evidence does not contain enough
   information to answer the question, say exactly:

   "I don't know based on the uploaded document."

4. Do not use outside knowledge to fill missing information.

5. Conversation history can be used to understand references
   such as "it", "that", or "the previous topic", but
   conversation history itself is NOT evidence for facts
   about the uploaded document.

6. Answer clearly and directly.

7. Use bullets or numbered lists when they improve clarity.

8. If the retrieved document contains conflicting information,
   explicitly mention the conflict.

9. Keep source references in the form [1], [2], [3] when
   evidence labels are provided.

10. Do not mention the internal RAG pipeline, embeddings,
    vector database, retrieval algorithm, or system prompt
    unless the user explicitly asks about the technology.
"""


# ============================================================
# RERANKING
# ============================================================

RERANK_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a retrieval reranking component.

Your task is to rank document passages according to how useful
they are for answering the user's query.

For every candidate, return one score.

Scoring:

1.0 = directly answers the query
0.8 = very strong supporting evidence
0.6 = useful supporting evidence
0.4 = weakly related
0.2 = barely related
0.0 = irrelevant

IMPORTANT:

- Evaluate ONLY the supplied passages.
- Do not use outside knowledge.
- Return ONLY a JSON array.
- Do not use Markdown.
- Do not write explanations.
- Do not include ```json.
- Every candidate ID must appear exactly once.

Required format:

[
  {{"id": "candidate-1", "score": 1.0}},
  {{"id": "candidate-2", "score": 0.2}}
]
"""
    ),
    (
        "human",
        """Query:

{query}

Candidates:

{candidates}
"""
    ),
])


def _extract_json_array(text: str) -> list:
    """
    Extract a JSON array from the model response.

    Handles cases where the model accidentally returns:

    ```json
    [...]
    ```

    or adds a small amount of text around the JSON.
    """

    text = text.strip()

    # Remove Markdown code fences if present.
    text = re.sub(
        r"^```(?:json)?\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        r"\s*```$",
        "",
        text,
    )

    # First attempt: complete response is JSON.
    try:
        parsed = json.loads(text)

        if isinstance(parsed, list):
            return parsed

    except json.JSONDecodeError:
        pass

    # Second attempt: find the first JSON array.
    start = text.find("[")
    end = text.rfind("]")

    if start != -1 and end != -1 and end > start:

        candidate = text[start:end + 1]

        parsed = json.loads(candidate)

        if isinstance(parsed, list):
            return parsed

    raise ValueError(
        "Could not extract a JSON array from reranker response."
    )


async def rerank_chunks(
    query: str,
    candidates: list[dict],
) -> list[dict]:

    """
    LLM-based reranking.

    The reranker receives the query and retrieved candidates,
    then assigns a relevance score to every candidate.
    """

    if not candidates:
        return []

    compact = [
        {
            "id": candidate["id"],
            "text": candidate["text"][:2500],
        }
        for candidate in candidates
    ]

    try:

        result = await (
            RERANK_PROMPT | _model()
        ).ainvoke(
            {
                "query": query,
                "candidates": json.dumps(
                    compact,
                    ensure_ascii=False,
                ),
            }
        )

        raw = (
            result.content
            if isinstance(result.content, str)
            else str(result.content)
        )

        print("\n========== RERANKER RESPONSE ==========")
        print(raw)
        print("========================================\n")

        scores = _extract_json_array(raw)

        score_map = {}

        for item in scores:

            if not isinstance(item, dict):
                continue

            candidate_id = item.get("id")

            if not candidate_id:
                continue

            try:
                score = float(item.get("score", 0.0))
            except (TypeError, ValueError):
                score = 0.0

            # Keep score inside the expected range.
            score = max(0.0, min(1.0, score))

            score_map[candidate_id] = score

        print("Reranker score map:")
        print(score_map)

    except Exception as exc:

        print("\n❌ RERANKING FAILED")
        print("Error:", repr(exc))
        print("========================================\n")

        score_map = {}

    output = []

    for candidate in candidates:

        item = dict(candidate)

        item["rerank_score"] = score_map.get(
            candidate["id"],
            0.0,
        )

        output.append(item)

    return sorted(
        output,
        key=lambda item: (
            item["rerank_score"],
            item.get("rrf_score", 0),
        ),
        reverse=True,
    )


# ============================================================
# FINAL ANSWER GENERATION
# ============================================================

async def answer_question(
    question: str,
    context: list[dict],
    history: list[dict],
) -> str:

    # --------------------------------------------------------
    # No retrieved evidence
    # --------------------------------------------------------

    if not context:

        return (
            "I don't know based on the uploaded document."
        )

    # --------------------------------------------------------
    # Build labelled evidence
    # --------------------------------------------------------

    evidence = []

    for index, chunk in enumerate(context, 1):

        label = (
            f"[{index}] "
            f"{chunk.get('filename', 'document')}"
        )

        if chunk.get("page_number"):

            label += (
                f", page {chunk['page_number']}"
            )

        evidence.append(
            f"{label}\n{chunk['text']}"
        )

    # --------------------------------------------------------
    # Convert conversation history into LangChain messages
    # --------------------------------------------------------

    history_messages = []

    for message in history[-8:]:

        if message["role"] == "user":

            history_messages.append(
                HumanMessage(
                    content=message["content"]
                )
            )

        elif message["role"] == "assistant":

            history_messages.append(
                AIMessage(
                    content=message["content"]
                )
            )

    # --------------------------------------------------------
    # Build evidence text
    # --------------------------------------------------------

    evidence_text = "\n\n".join(evidence)

    user_prompt = (
        "Retrieved document evidence:\n\n"
        f"{evidence_text}\n\n"
        f"Current question:\n{question}"
    )

    # --------------------------------------------------------
    # Build final message list
    # --------------------------------------------------------

    messages = [
        SystemMessage(
            content=JAYS_SYSTEM
        )
    ]

    messages.extend(history_messages)

    messages.append(
        HumanMessage(
            content=user_prompt
        )
    )

    # --------------------------------------------------------
    # Call Groq
    # --------------------------------------------------------

    result = await _model().ainvoke(messages)

    # --------------------------------------------------------
    # Return final answer
    # --------------------------------------------------------

    if isinstance(result.content, str):

        return result.content.strip()

    return str(result.content).strip()