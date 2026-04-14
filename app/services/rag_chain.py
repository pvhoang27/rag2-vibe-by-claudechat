"""
app/services/rag_chain.py
--------------------------
LangChain RAG pipeline using Ollama local LLM.
"""

from __future__ import annotations

import time
from typing import Optional

from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from app.core.config import get_settings
from app.core.logger import logger
from app.models.schemas import ChatResponse, Source
from app.services.vector_store import VectorStoreService


RAG_PROMPT_TEMPLATE = """Bạn là một trợ lý AI hữu ích và chuyên nghiệp.
Hãy trả lời câu hỏi dựa trên các đoạn văn bản được cung cấp bên dưới.
Nếu thông tin không có trong context, hãy nói thẳng rằng bạn không tìm thấy thông tin liên quan.
Không được bịa đặt hay suy đoán ngoài context.

Context:
{context}

Câu hỏi: {question}

Trả lời:"""


class RAGChainService:
    """Manages the RAG question-answering pipeline."""

    def __init__(self, vector_store: VectorStoreService) -> None:
        self._settings = get_settings()
        self._vector_store = vector_store
        self._llm = OllamaLLM(
            base_url=self._settings.ollama_base_url,
            model=self._settings.ollama_llm_model,
            temperature=0.1,
        )
        self._prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=RAG_PROMPT_TEMPLATE,
        )
        self._chain = self._prompt | self._llm | StrOutputParser()

    def query(
        self,
        question: str,
        top_k: Optional[int] = None,
    ) -> ChatResponse:
        """Run a RAG query and return structured response."""
        k = top_k or self._settings.top_k_retrieval
        t0 = time.perf_counter()

        # 1. Retrieve relevant chunks
        results = self._vector_store.similarity_search(question, k=k)

        # 2. Build context string
        if results:
            context = "\n\n---\n\n".join(doc.page_content for doc, _ in results)
        else:
            context = "Không tìm thấy thông tin liên quan trong tài liệu."

        # 3. Generate answer
        answer = self._chain.invoke({"context": context, "question": question})

        latency_ms = (time.perf_counter() - t0) * 1000
        logger.info(
            f"Query answered in {latency_ms:.0f}ms | "
            f"chunks_retrieved={len(results)} | model={self._settings.ollama_llm_model}"
        )

        # 4. Build source list
        sources = []
        for doc, score in results:
            score_value = float(score) if score is not None else None
            # Some vector backends may return non-normalized scores; hide them from UI.
            if score_value is not None and not (0.0 <= score_value <= 1.0):
                score_value = None

            sources.append(
                Source(
                    content=doc.page_content[:300],
                    source=doc.metadata.get("source") or "unknown",
                    page=doc.metadata.get("page"),
                    score=round(score_value, 4) if score_value is not None else None,
                )
            )

        return ChatResponse(
            answer=answer,
            sources=sources,
            model=self._settings.ollama_llm_model,
            latency_ms=round(latency_ms, 2),
        )

    def get_context_for_eval(self, question: str, top_k: int = 3) -> list[str]:
        """Return raw context strings — used by evaluation service."""
        results = self._vector_store.similarity_search(question, k=top_k)
        return [doc.page_content for doc, _ in results]
