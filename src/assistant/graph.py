"""LangGraph workflow for the medical assistant."""
from __future__ import annotations

from pathlib import Path
from typing import Any, TypedDict

from langchain_core.documents import Document
from langgraph.graph import END, StateGraph

from src.assistant import audit_logger, explainer, guardrails, patient_store
from src.assistant.chain import build_generation_prompt, format_kb_context
from src.assistant.local_llm import (
    LocalAssistantLLM,
    LocalModelInitializationError,
    LocalModelOutputError,
    allow_fallback,
)
from src.assistant.retriever import get_retriever
from src.config import ASSISTANT_LOG_PATH, VECTORSTORE_DIR
from src.llm.client import get_llm_client
from src.logging_utils import log_event, setup_json_logger
from src.multimodal.audio import analyze_transcript, transcribe
from src.multimodal.video import analyze_video, generate_video_report

_CLASSIFY_PROMPT = (
    "Classify the following question as either 'medical' (related to medicine, "
    "oncology, cancer, diagnosis, treatment, symptoms, anatomy, or clinical topics) "
    "or 'out_of_scope' (anything else).\n"
    "Reply with ONLY the single word: medical OR out_of_scope.\n\n"
    "Question: {query}"
)

DEFAULT_YOLO_MODEL_PATH = (
    Path(__file__).resolve().parents[2]
    / "artifacts"
    / "yolo"
    / "bleeding_yolov8n"
    / "weights"
    / "best.pt"
)


class AssistantState(TypedDict):
    query: str
    patient_id: str | None
    intent: str
    retrieved_docs: list[Document]
    kb_context: str
    patient_record: dict[str, Any] | None
    patient_context: str
    answer: str
    kb_sources: list[str]
    patient_source: str | None
    used_patient_context: bool
    refused: bool
    error: str | None
    audio_path: str | None
    audio_transcript: str | None
    audio_analysis: str | None
    video_path: str | None
    video_report: str | None


def _build_classifier() -> Any:
    llm_client = get_llm_client()
    if llm_client is None:
        return None
    return llm_client


def _build_generator() -> tuple[LocalAssistantLLM, Any | None]:
    return LocalAssistantLLM(), get_llm_client() if allow_fallback() else None


_MEDICAL_KEYWORDS = (
    "cancer", "oncology", "tumor", "tumour", "diagnosis", "diagnostic",
    "treatment", "biopsy", "breast", "malignan", "bi-rads", "birads",
    "radiology", "imaging", "lesion", "lump", "lymph", "chemo", "radiation",
    "pathology", "histology", "staging", "metastas", "carcinoma", "sarcoma",
    "prognosis", "surgery", "hormone", "receptor", "mammogram", "ultrasound",
    "mri", "pet scan", "biopsia", "tumor", "câncer", "cancer", "oncologia",
    "diagnóstico", "tratamento", "mama", "linfonodo", "quimio", "radioterapia",
)


def _keyword_classify(query: str) -> str:
    text = query.lower()
    return "medical" if any(k in text for k in _MEDICAL_KEYWORDS) else "out_of_scope"


def _classify_query(query: str, classifier: Any | None) -> str:
    if classifier is None:
        return _keyword_classify(query)

    try:
        result = classifier.generate(_CLASSIFY_PROMPT.format(query=query), temperature=0.0, max_tokens=16)
        text = result.text.strip().lower()
        return "medical" if text == "medical" or text.startswith("medical") else "out_of_scope"
    except Exception:
        # Gemini classifier unavailable — fall back to keyword matching
        return _keyword_classify(query)


def _generate_with_policy(prompt: str, local_llm: LocalAssistantLLM, fallback_llm: Any | None) -> str:
    try:
        return local_llm.generate(prompt)
    except (LocalModelInitializationError, LocalModelOutputError):
        if fallback_llm is None:
            raise
        result = fallback_llm.generate(prompt, temperature=0.2, max_tokens=512)
        return result.text.strip()


def build_graph():
    """Build and compile the LangGraph medical assistant workflow."""
    logger = setup_json_logger("assistant", ASSISTANT_LOG_PATH)
    retriever = get_retriever(VECTORSTORE_DIR)
    classifier = _build_classifier()
    local_generator, fallback_generator = _build_generator()

    def classify_intent(state: AssistantState) -> AssistantState:
        is_blocked, refusal_message = guardrails.filter_input(state["query"])
        if is_blocked:
            return {
                **state,
                "intent": "out_of_scope",
                "answer": refusal_message or guardrails._INPUT_REFUSAL,
                "error": None,
            }
        intent = _classify_query(state["query"], classifier)
        return {**state, "intent": intent, "error": None}

    def retrieve_kb_context(state: AssistantState) -> AssistantState:
        try:
            docs = retriever.invoke(state["query"])
        except Exception as exc:
            return {**state, "retrieved_docs": [], "kb_context": "", "error": str(exc)}
        return {**state, "retrieved_docs": docs, "kb_context": format_kb_context(docs)}

    def retrieve_patient_context(state: AssistantState) -> AssistantState:
        patient_id = state.get("patient_id")
        if not patient_id:
            return {
                **state,
                "patient_record": None,
                "patient_context": "No patient-specific context provided.",
                "patient_source": None,
                "used_patient_context": False,
            }

        record = patient_store.load_patient_record(patient_id)
        if record is None:
            return {
                **state,
                "patient_record": None,
                "patient_context": "",
                "patient_source": None,
                "used_patient_context": False,
                "error": f"Patient record not found for patient_id={patient_id}.",
                "answer": f"Patient record not found for patient_id={patient_id}.",
            }

        return {
            **state,
            "patient_record": record,
            "patient_context": patient_store.format_patient_context(record),
            "patient_source": patient_store.get_patient_source(record),
            "used_patient_context": True,
        }

    def process_audio(state: AssistantState) -> AssistantState:
        audio_path = state.get("audio_path")
        if not audio_path:
            return state

        try:
            transcript = transcribe(audio_path)
            analysis = analyze_transcript(transcript)
        except Exception as exc:
            return {
                **state,
                "error": str(exc),
                "answer": f"Unable to process audio input: {exc}",
            }

        return {
            **state,
            "audio_transcript": transcript,
            "audio_analysis": analysis,
        }

    def process_video(state: AssistantState) -> AssistantState:
        video_path = state.get("video_path")
        if not video_path:
            return state

        try:
            analysis = analyze_video(video_path, DEFAULT_YOLO_MODEL_PATH)
            report = generate_video_report(analysis.detections)
        except Exception as exc:
            return {
                **state,
                "error": str(exc),
                "answer": f"Unable to process video input: {exc}",
            }

        return {**state, "video_report": report}

    def generate_response(state: AssistantState) -> AssistantState:
        if state.get("error"):
            return state
        prompt = build_generation_prompt(
            question=state["query"],
            patient_context=state["patient_context"],
            kb_context=state["kb_context"],
            audio_analysis=state.get("audio_analysis"),
            video_report=state.get("video_report"),
        )
        try:
            answer = _generate_with_policy(prompt, local_generator, fallback_generator)
        except Exception as exc:
            return {**state, "error": str(exc), "answer": "Unable to generate a response at this time."}
        return {**state, "answer": answer}

    def validate_response(state: AssistantState) -> AssistantState:
        if state.get("error"):
            return state

        kb_sources = sorted(
            {
                Path(doc.metadata["source"]).name if "source" in doc.metadata else "unknown"
                for doc in state.get("retrieved_docs", [])
            }
        )
        patient_source = state.get("patient_source") if state.get("used_patient_context") else None
        answer = guardrails.filter_output(
            state["answer"],
            has_context=bool(kb_sources or patient_source),
        )
        explanation = explainer.explain_prediction(
            query=state["query"],
            response=answer,
            sources=kb_sources,
            patient_source=patient_source,
            ml_context=(state.get("patient_record") or {}).get("ml_context"),
        )
        if explanation:
            answer += f"\n\nWhy this answer: {explanation}"
        if patient_source:
            answer += f"\n\nClinical Context Source: {patient_source}"
        if kb_sources:
            answer += f"\nKnowledge Base Sources: {', '.join(kb_sources)}"

        audit_logger.log_interaction(
            user_query=state["query"],
            retrieved_docs=kb_sources,
            model_response=answer,
            guardrail_triggered=guardrails.has_guardrail_signal(state["answer"], answer),
            intent=state["intent"],
            error=state.get("error"),
            patient_id=state.get("patient_id"),
            patient_context_used=bool(patient_source),
            kb_sources=kb_sources,
            patient_source=patient_source,
        )
        log_event(
            logger,
            "interaction",
            query=state["query"],
            intent=state["intent"],
            kb_sources=kb_sources,
            patient_source=patient_source,
            answer_len=len(answer),
            refused=False,
        )
        return {
            **state,
            "answer": answer,
            "kb_sources": kb_sources,
            "patient_source": patient_source,
            "refused": False,
        }

    def refuse_response(state: AssistantState) -> AssistantState:
        final_answer = state.get("answer") or guardrails._INPUT_REFUSAL
        audit_logger.log_interaction(
            user_query=state["query"],
            retrieved_docs=[],
            model_response=final_answer,
            guardrail_triggered=True,
            intent=state["intent"],
            error=state.get("error"),
            patient_id=state.get("patient_id"),
            patient_context_used=False,
            kb_sources=[],
            patient_source=None,
        )
        log_event(
            logger,
            "interaction",
            query=state["query"],
            intent=state["intent"],
            kb_sources=[],
            patient_source=None,
            answer_len=len(final_answer),
            refused=True,
            error=state.get("error"),
        )
        return {
            **state,
            "answer": final_answer,
            "kb_sources": [],
            "patient_source": None,
            "used_patient_context": False,
            "refused": True,
        }

    def error_response(state: AssistantState) -> AssistantState:
        final_answer = state.get("answer") or "Unable to complete the request."
        audit_logger.log_interaction(
            user_query=state["query"],
            retrieved_docs=[],
            model_response=final_answer,
            guardrail_triggered=False,
            intent=state["intent"],
            error=state.get("error"),
            patient_id=state.get("patient_id"),
            patient_context_used=False,
            kb_sources=[],
            patient_source=None,
        )
        return {
            **state,
            "answer": final_answer,
            "kb_sources": [],
            "patient_source": None,
            "used_patient_context": False,
            "refused": False,
        }

    def route_after_classify(state: AssistantState) -> str:
        if state.get("intent") == "medical":
            return "retrieve_kb_context"
        return "refuse_response"

    def route_after_patient_context(state: AssistantState) -> str:
        if state.get("error"):
            return "error_response"
        return "process_audio"

    def route_after_audio(state: AssistantState) -> str:
        if state.get("error"):
            return "error_response"
        return "process_video"

    def route_after_video(state: AssistantState) -> str:
        if state.get("error"):
            return "error_response"
        return "generate_response"

    graph = StateGraph(AssistantState)
    graph.add_node("classify_intent", classify_intent)
    graph.add_node("retrieve_kb_context", retrieve_kb_context)
    graph.add_node("retrieve_patient_context", retrieve_patient_context)
    graph.add_node("process_audio", process_audio)
    graph.add_node("process_video", process_video)
    graph.add_node("generate_response", generate_response)
    graph.add_node("validate_response", validate_response)
    graph.add_node("refuse_response", refuse_response)
    graph.add_node("error_response", error_response)

    graph.set_entry_point("classify_intent")
    graph.add_conditional_edges("classify_intent", route_after_classify)
    graph.add_edge("retrieve_kb_context", "retrieve_patient_context")
    graph.add_conditional_edges("retrieve_patient_context", route_after_patient_context)
    graph.add_conditional_edges("process_audio", route_after_audio)
    graph.add_conditional_edges("process_video", route_after_video)
    graph.add_edge("generate_response", "validate_response")
    graph.add_edge("validate_response", END)
    graph.add_edge("refuse_response", END)
    graph.add_edge("error_response", END)

    return graph.compile()
