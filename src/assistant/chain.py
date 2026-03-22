"""LangChain LCEL chain for the medical assistant (M3).

Combines the retriever with ChatGoogleGenerativeAI to generate
grounded responses from retrieved context.
"""
from __future__ import annotations

from operator import itemgetter

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableParallel


_PROMPT_TEMPLATE = ChatPromptTemplate.from_template(
    """You are a medical assistant helping doctors with clinical questions about oncology.
Use ONLY the retrieved knowledge below to answer. If the context does not contain \
enough information to answer, say so clearly. Never prescribe medications or make \
definitive diagnoses — always recommend consulting a qualified physician.

{ml_context_block}
Retrieved Knowledge:
{context}

Question: {question}

Answer in the same language as the question. Be concise and cite relevant details \
from the retrieved context."""
)


def _format_docs(docs) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


def _format_ml_context(ml_context: str) -> str:
    if not ml_context:
        return ""
    return f"ML Pipeline Context:\n{ml_context}\n"


def build_chain(retriever, llm):
    """Build a RAG chain that accepts a dict input.

    Input schema: {"question": str, "ml_context": str | None}
    Output: str (the generated answer)

    None → "" coercion for ml_context happens inside this function.
    """
    chain = (
        RunnableParallel(
            context=(itemgetter("question") | retriever | _format_docs),
            question=itemgetter("question"),
            ml_context_block=RunnableLambda(
                lambda x: _format_ml_context(x.get("ml_context") or "")
            ),
        )
        | _PROMPT_TEMPLATE
        | llm
        | StrOutputParser()
    )
    return chain
