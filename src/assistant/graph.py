"""LangGraph StateGraph for the medical assistant workflow (M3).

Nodes: classify_intent → retrieve_context → generate_response → validate_response
       classify_intent → refuse_response  (out-of-scope branch)
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import TypedDict

from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, StateGraph

from src.assistant import audit_logger, explainer, guardrails
from src.assistant.chain import build_chain
from src.assistant.guardrails import _DISCLAIMER
from src.assistant.retriever import get_retriever
from src.config import ASSISTANT_LOG_PATH, VECTORSTORE_DIR
from src.logging_utils import log_event, setup_json_logger

_REFUSAL_MSG = (
    "Posso responder apenas perguntas médicas relacionadas a oncologia e diagnóstico. "
    "Por favor, reformule sua pergunta dentro desse escopo."
)
_CLASSIFY_PROMPT = (
    "Classify the following question as either 'medical' (related to medicine, "
    "oncology, cancer, diagnosis, treatment, symptoms, anatomy, or clinical topics) "
    "or 'out_of_scope' (anything else).\n"
    "Reply with ONLY the single word: medical OR out_of_scope\n\n"
    "Question: {query}"
)


class AssistantState(TypedDict):
    query: str
    intent: str                    # "medical" | "out_of_scope"
    retrieved_docs: list[Document]
    ml_context: str | None
    response: str
    sources: list[str]
    refused: bool
    error: str | None


def _get_llm() -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model=os.getenv("GEMINI_MODEL", "gemini-2.0-flash"),
        google_api_key=os.getenv("GEMINI_API_KEY"),
        temperature=0.2,
    )


def build_graph():
    """Build and compile the LangGraph medical assistant StateGraph.

    Returns:
        CompiledGraph ready for .invoke()
    """
    logger = setup_json_logger("assistant", ASSISTANT_LOG_PATH)
    retriever = get_retriever(VECTORSTORE_DIR)
    llm = _get_llm()
    chain = build_chain(retriever, llm)

    # ── Node functions ────────────────────────────────────────────

    def classify_intent(state: AssistantState) -> AssistantState:
        is_blocked, refusal_message = guardrails.filter_input(state["query"])
        if is_blocked:
            return {**state, "intent": "out_of_scope", "response": refusal_message, "error": None}
        try:
            prompt = _CLASSIFY_PROMPT.format(query=state["query"])
            result = llm.invoke(prompt)
            text = result.content.strip().lower()
            intent = "medical" if text == "medical" or text.startswith("medical") else "out_of_scope"
        except Exception as exc:
            intent = "out_of_scope"
            return {**state, "intent": intent, "error": str(exc)}
        return {**state, "intent": intent, "error": None}

    def retrieve_context(state: AssistantState) -> AssistantState:
        try:
            docs = retriever.invoke(state["query"])
        except Exception as exc:
            return {**state, "retrieved_docs": [], "error": str(exc)}
        return {**state, "retrieved_docs": docs}

    def generate_response(state: AssistantState) -> AssistantState:
        answer = chain.invoke(
            {"question": state["query"], "ml_context": state["ml_context"]}
        )
        return {**state, "response": answer}

    def validate_response(state: AssistantState) -> AssistantState:
        sources = [
            Path(doc.metadata["source"]).name if "source" in doc.metadata else "unknown"
            for doc in state.get("retrieved_docs", [])
        ]
        original_response = state["response"]
        response = guardrails.filter_output(original_response)
        output_guardrail_fired = response != original_response
        explanation = explainer.explain_prediction(
            query=state["query"],
            response=response,
            sources=sources,
            ml_context=state["ml_context"],
        )
        if explanation:
            # _DISCLAIMER imported from guardrails — single source of truth
            response = response.replace(
                _DISCLAIMER,
                f"\n\nExplanation: {explanation}{_DISCLAIMER}",
            )
        audit_logger.log_interaction(
            user_query=state["query"],
            retrieved_docs=sources,
            model_response=response,
            guardrail_triggered=output_guardrail_fired,
            intent=state["intent"],
            error=state.get("error"),
        )
        log_event(
            logger,
            "interaction",
            query=state["query"],
            intent=state["intent"],
            sources=sources,
            response_len=len(response),
            refused=False,
        )
        return {**state, "response": response, "sources": sources, "refused": False}

    def refuse_response(state: AssistantState) -> AssistantState:
        final_response = state.get("response") or _REFUSAL_MSG
        audit_logger.log_interaction(
            user_query=state["query"],
            retrieved_docs=[],
            model_response=final_response,
            guardrail_triggered=True,
            intent=state["intent"],
            error=state.get("error"),
        )
        log_event(
            logger,
            "interaction",
            query=state["query"],
            intent=state["intent"],
            sources=[],
            response_len=len(final_response),
            refused=True,
            error=state.get("error"),
        )
        return {**state, "response": final_response, "sources": [], "refused": True}

    def route_intent(state: AssistantState) -> str:
        if state.get("intent") == "medical":
            return "retrieve_context"
        return "refuse_response"

    # ── Build graph ───────────────────────────────────────────────

    graph = StateGraph(AssistantState)
    graph.add_node("classify_intent", classify_intent)
    graph.add_node("retrieve_context", retrieve_context)
    graph.add_node("generate_response", generate_response)
    graph.add_node("validate_response", validate_response)
    graph.add_node("refuse_response", refuse_response)

    graph.set_entry_point("classify_intent")
    graph.add_conditional_edges("classify_intent", route_intent)
    graph.add_edge("retrieve_context", "generate_response")
    graph.add_edge("generate_response", "validate_response")
    graph.add_edge("validate_response", END)
    graph.add_edge("refuse_response", END)

    return graph.compile()
