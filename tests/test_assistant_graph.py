from types import SimpleNamespace

import pytest

langgraph = pytest.importorskip("langgraph")
langchain_core = pytest.importorskip("langchain_core")

from langchain_core.documents import Document

from src.assistant import graph as graph_mod
from src.assistant.chain import build_generation_prompt


class _FakeRetriever:
    def invoke(self, query):
        return [
            Document(page_content="Breast cancer symptoms can include a lump.", metadata={"source": "/tmp/kb1.txt"}),
            Document(page_content="Biopsy is recommended for suspicious lesions.", metadata={"source": "/tmp/kb2.txt"}),
        ]


class _FakeLocalLLM:
    prompts = []

    def generate(self, prompt, max_new_tokens=256):
        self.prompts.append(prompt)
        return "The patient should undergo physician-guided diagnostic evaluation."


class _FakeClassifier:
    def generate(self, prompt, temperature=0.0, max_tokens=16):
        class _Result:
            text = "medical"
        return _Result()


@pytest.fixture
def fake_graph(monkeypatch):
    _FakeLocalLLM.prompts = []
    monkeypatch.setattr(graph_mod, "get_retriever", lambda *_args, **_kwargs: _FakeRetriever())
    monkeypatch.setattr(graph_mod, "_build_classifier", lambda: _FakeClassifier())
    monkeypatch.setattr(graph_mod, "_build_generator", lambda: (_FakeLocalLLM(), None))
    return graph_mod.build_graph()


def _initial_state(**overrides):
    state = {
        "query": "What should be reviewed for this patient?",
        "patient_id": "P-0001",
        "intent": "",
        "retrieved_docs": [],
        "kb_context": "",
        "patient_record": None,
        "patient_context": "",
        "answer": "",
        "kb_sources": [],
        "patient_source": None,
        "used_patient_context": False,
        "refused": False,
        "error": None,
        "audio_path": None,
        "audio_transcript": None,
        "audio_analysis": None,
        "video_path": None,
        "video_report": None,
    }
    state.update(overrides)
    return state


def test_graph_returns_kb_and_patient_sources(fake_graph):
    result = fake_graph.invoke(_initial_state())
    assert result["patient_source"] == "patient_record:P-0001"
    assert result["kb_sources"] == ["kb1.txt", "kb2.txt"]
    assert "Clinical Context Source: patient_record:P-0001" in result["answer"]
    assert "Knowledge Base Sources: kb1.txt, kb2.txt" in result["answer"]


def test_graph_missing_patient_returns_error(fake_graph):
    result = fake_graph.invoke(_initial_state(patient_id="P-9999"))
    assert "Patient record not found" in result["answer"]


def test_graph_skips_multimodal_nodes_without_paths(fake_graph, monkeypatch):
    monkeypatch.setattr(
        graph_mod,
        "transcribe",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("unexpected audio call")),
    )
    monkeypatch.setattr(
        graph_mod,
        "analyze_video",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("unexpected video call")),
    )

    result = fake_graph.invoke(_initial_state(audio_path=None, video_path=None))

    assert result["audio_transcript"] is None
    assert result["audio_analysis"] is None
    assert result["video_report"] is None


def test_graph_processes_audio_and_video_before_generation(fake_graph, monkeypatch):
    calls = {}

    def _fake_transcribe(audio_path):
        calls["audio_path"] = audio_path
        return "transcript text"

    def _fake_analyze_transcript(transcript):
        calls["transcript"] = transcript
        return "audio clinical report"

    def _fake_analyze_video(video_path, model_path):
        calls["video_path"] = video_path
        calls["model_path"] = model_path
        return SimpleNamespace(detections=["detection"])

    def _fake_generate_video_report(detections):
        calls["detections"] = detections
        return "video clinical report"

    monkeypatch.setattr(graph_mod, "transcribe", _fake_transcribe)
    monkeypatch.setattr(graph_mod, "analyze_transcript", _fake_analyze_transcript)
    monkeypatch.setattr(graph_mod, "analyze_video", _fake_analyze_video)
    monkeypatch.setattr(graph_mod, "generate_video_report", _fake_generate_video_report)

    result = fake_graph.invoke(_initial_state(audio_path="sample.wav", video_path="sample.mp4"))

    assert calls["audio_path"] == "sample.wav"
    assert calls["transcript"] == "transcript text"
    assert calls["video_path"] == "sample.mp4"
    assert calls["model_path"] == graph_mod.DEFAULT_YOLO_MODEL_PATH
    assert calls["detections"] == ["detection"]
    assert result["audio_transcript"] == "transcript text"
    assert result["audio_analysis"] == "audio clinical report"
    assert result["video_report"] == "video clinical report"
    assert "Audio clinical analysis:\naudio clinical report" in _FakeLocalLLM.prompts[-1]
    assert "Video clinical report:\nvideo clinical report" in _FakeLocalLLM.prompts[-1]


def test_build_generation_prompt_includes_multimodal_context():
    prompt = build_generation_prompt(
        question="What should be reviewed?",
        patient_context="No patient-specific context provided.",
        kb_context="No relevant knowledge base context found.",
        audio_analysis="audio report",
        video_report="video report",
    )

    assert "Audio clinical analysis:\naudio report" in prompt
    assert "Video clinical report:\nvideo report" in prompt
