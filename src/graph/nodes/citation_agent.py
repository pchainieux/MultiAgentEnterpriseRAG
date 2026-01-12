from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from src.graph.state import GraphState
from src.rag.llm.models import get_citation_llm
from src.rag.llm.prompts import CITATION_SYSTEM_PROMPT

_MAX_DOCS_IN_CONTEXT = 10
_SNIPPET_CHARS = 240

def _build_citation_context(documents: List[Document]) -> str:
    """
    Build a compact numbered context block from retrieved documents so the citation LLM can reference evidence by passage index.
    Inputs: documents retrieved chunks with metadata; Outputs: a formatted string containing up to _MAX_DOCS_IN_CONTEXT numbered passages and key metadata.
    """
    lines: List[str] = []
    for i, d in enumerate(documents[:_MAX_DOCS_IN_CONTEXT]):
        meta = d.metadata or {}
        src = meta.get("source", "unknown")
        src_name = meta.get("source_name", Path(src).name if src != "unknown" else "unknown")
        page = meta.get("page", meta.get("section", ""))

        lines.append(
            f"[{i}] doc_id={meta.get('doc_id', meta.get('id', ''))} "
            f"source={src} source_name={src_name} page={page}\n"
            f"{(d.page_content or '')[:800]}"
        )
    return "\n\n".join(lines)


def _extract_json(text: str) -> str:
    """
    Normalize an LLM response into raw JSON by stripping optional triple backtick code fences etc...
    Inputs: model output that may include ``` fences; Outputs: a string expected to be valid JSON.
    """
    s = (text or "").strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z]*\n", "", s)
        s = re.sub(r"\n```$", "", s).strip()
    return s


def _try_parse_json(resp_text: str) -> Optional[Dict[str, Any]]:
    """
    Try to parse a model response as JSON, returning None if parsing fails to enable retry/refusal behavior upstream.
    Inputs: raw model output; Outputs: a parsed dict on success, otherwise None.
    """
    try:
        return json.loads(_extract_json(resp_text))
    except Exception:
        return None


def _coerce_int(value: Any) -> Optional[int]:
    """
    Convert a citation index value into an integer when possible to robustly handle model outputs that might emit numbers as strings.
    Inputs: value ; Outputs: an int if coercion succeeds, otherwise None.
    """
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value.strip())
        except Exception:
            return None
    return None


def _normalize_and_enrich_citations(
    citations: Any, documents: List[Document]
) -> List[Dict[str, Any]]:
    """
    Validate citations returned by the model and enrich them with canonical metadata and a text snippet for the UI.
    Inputs: model-provided citation list and documents used as evidence; Outputs: a list of normalised citation dicts.
    """
    if not isinstance(citations, list):
        return []

    out: List[Dict[str, Any]] = []
    max_idx = min(len(documents), _MAX_DOCS_IN_CONTEXT) - 1
    if max_idx < 0:
        return []

    for c in citations:
        if not isinstance(c, dict):
            continue

        idx = _coerce_int(c.get("index"))
        if idx is None or idx < 0 or idx > max_idx:
            continue

        d = documents[idx]
        meta = d.metadata or {}

        source = meta.get("source", "unknown")
        source_name = meta.get(
            "source_name",
            Path(source).name if isinstance(source, str) and source != "unknown" else "unknown",
        )
        page = meta.get("page", meta.get("section", ""))
        doc_id = meta.get("doc_id", meta.get("id", ""))

        out.append(
            {
                "index": idx,
                "doc_id": doc_id,
                "source": source,
                "source_name": source_name,
                "page": page,
                "snippet": (d.page_content or "")[:_SNIPPET_CHARS],
            }
        )

    return out


def citation_node(state: GraphState) -> dict:
    """
    Post process the draft answer by adding inline numeric citation markers and emitting structured citations, refusing when evidence is missing or citations are invalid.
    Inputs: state ; Outputs: a partial state update dict with 'answer', 'citations', and an internal trace message.
    """
    llm = get_citation_llm()

    answer = (state.get("answer") or "").strip()
    documents: List[Document] = state.get("documents", [])

    if not answer:
        return {"citations": []}

    if not documents:
        refused = "I don't know based on the available documents."
        msg = AIMessage(
            content="CitationAgent: no documents available; refusing.",
            name="citation_agent",
        )
        return {"answer": refused, "citations": [], "messages": [msg]}

    context_block = _build_citation_context(documents)

    prompt_messages = [
        SystemMessage(content=CITATION_SYSTEM_PROMPT),
        HumanMessage(
            content=(
                "Existing answer:\n"
                f"{answer}\n\n"
                "Candidate supporting documents:\n"
                f"{context_block}\n\n"
                "Return JSON with 'answer_with_citations' and 'citations'."
            )
        ),
    ]

    parsed: Optional[Dict[str, Any]] = None
    for _ in range(2):
        resp = llm.invoke(prompt_messages)
        parsed = _try_parse_json(resp.content)
        if parsed is not None:
            break
        prompt_messages.append(
            HumanMessage(
                content=(
                    "Your previous output was not valid JSON. "
                    "Return ONLY valid JSON, no prose, no code fences."
                )
            )
        )

    if parsed is None:
        refused = "I can't provide a properly cited answer with the current evidence."
        msg = AIMessage(
            content="CitationAgent: invalid JSON after retry; refusing.",
            name="citation_agent",
        )
        return {"answer": refused, "citations": [], "messages": [msg]}

    answer_with_citations = (parsed.get("answer_with_citations") or "").strip() or answer
    citations_struct = _normalize_and_enrich_citations(parsed.get("citations"), documents)

    if not citations_struct:
        refused = "I can't provide a properly cited answer with the current evidence."
        msg = AIMessage(
            content="CitationAgent: empty/invalid citations; refusing.",
            name="citation_agent",
        )
        return {"answer": refused, "citations": [], "messages": [msg]}

    msg = AIMessage(content="CitationAgent attached validated citations.", name="citation_agent")
    return {
        "answer": answer_with_citations,
        "citations": citations_struct,
        "messages": [msg],
    }
