import pytest

langgraph = pytest.importorskip("langgraph")
langchain_core = pytest.importorskip("langchain_core")

from langchain_core.documents import Document

from src.assistant import graph as graph_mod


class _FakeRetriever:
    def invoke(self, query):
        return [
            Document(page_content="Breast cancer symptoms can include a lump.", metadata={"source": "/tmp/kb1.txt"}),
            Document(page_content="Biopsy is recommended for suspicious lesions.", metadata={"source": "/tmp/kb2.txt"}),
        ]


class _FakeLocalLLM:
    def generate(self, prompt, max_new_tokens=256):
        return "The patient should undergo physician-guided diagnostic evaluation."


class _FakeClassifier:
    def generate(self, prompt, temperature=0.0, max_tokens=16):
        class _Result:
            text = "medical"
        return _Result()


@pytest.fixture
def fake_graph(monkeypatch):
    monkeypatch.setattr(graph_mod, "get_retriever", lambda *_args, **_kwargs: _FakeRetriever())
    monkeypatch.setattr(graph_mod, "_build_classifier", lambda: _FakeClassifier())
    monkeypatch.setattr(graph_mod, "_build_generator", lambda: (_FakeLocalLLM(), None))
    return graph_mod.build_graph()


def test_graph_returns_kb_and_patient_sources(fake_graph):
    result = fake_graph.invoke(
        {
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
        }
    )
    assert result["patient_source"] == "patient_record:P-0001"
    assert result["kb_sources"] == ["kb1.txt", "kb2.txt"]
    assert "Clinical Context Source: patient_record:P-0001" in result["answer"]
    assert "Knowledge Base Sources: kb1.txt, kb2.txt" in result["answer"]


def test_graph_missing_patient_returns_error(fake_graph):
    result = fake_graph.invoke(
        {
            "query": "What should be reviewed for this patient?",
            "patient_id": "P-9999",
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
        }
    )
    assert "Patient record not found" in result["answer"]
