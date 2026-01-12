from __future__ import annotations

from typing import List

from openai import OpenAI
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from sentence_transformers import SentenceTransformer

from src.app.core.config import settings
from src.rag.llm.ollama_adapter import get_ollama_llm  

class OpenAIChatWrapper(BaseChatModel):
    """
    Minimal LangChain compatible wrapper around OpenAI Chat models.
    """

    def __init__(self, model_name: str, api_key: str | None):
        super().__init__()
        self._client = OpenAI(api_key=api_key)
        self.model_name = model_name

    @property
    def _llm_type(self) -> str:
        return "openai-chat"

    def _generate(self, messages: List[BaseMessage], stop=None, run_manager=None, **kwargs):
        """
        Convert LangChain messages into OpenAI chat completions format and request a completion, returning a LangChain LLMResult.
        Inputs: messages, stop/run_manager, kwargs (such as temperature); Outputs: LLMResult containing the generated assistant text.
        """
        prompt = []
        for m in messages:
            role = "user"
            if m.type == "system":
                role = "system"
            elif m.type == "ai":
                role = "assistant"
            prompt.append({"role": role, "content": m.content})

        resp = self._client.chat.completions.create(
            model=self.model_name,
            messages=prompt,
            temperature=kwargs.get("temperature", 0.1),
        )

        content = resp.choices[0].message.content or ""

        from langchain_core.outputs import Generation, LLMResult
        gen = Generation(text=content)
        return LLMResult(generations=[[gen]])

    def invoke(self, input, **kwargs):
        """
        Provide a simplified invoke() interface that returns an object exposing a 'content' attribute for compatibility with the rest of the codebase.
        Inputs: input (list[BaseMessage]) and optional kwargs; Outputs: a lightweight response object with a 'content' string field.
        """
        result = self._generate(input, **kwargs)
        return type("Resp", (), {"content": result.generations[0][0].text})


def _get_openai_llm() -> BaseChatModel:
    return OpenAIChatWrapper(
        model_name=settings.OPENAI_MODEL_NAME,
        api_key=settings.OPENAI_API_KEY,
    )

def _get_base_llm() -> BaseChatModel:
    """
    Route chat model construction to the configured provider.
    Inputs: none; Outputs: a BaseChatModel instance corresponding to the selected provider.
    """
    provider = (settings.LLM_PROVIDER or "openai").lower()
    if provider == "ollama":
        return get_ollama_llm()
    return _get_openai_llm()

# I left here the option to use different LLMs for different tasks
def get_planner_llm() -> BaseChatModel:
    return _get_base_llm()

def get_reasoning_llm() -> BaseChatModel:
    return _get_base_llm()

def get_citation_llm() -> BaseChatModel:
    return _get_base_llm()

class BGEEmbeddingModel:
    """
    Simple wrapper around SentenceTransformer for BGE/BGE-large.
    """

    def __init__(self, model_name: str):
        self._model = SentenceTransformer(model_name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a batch of texts into normalised dense vectors for ingestion and retrieval.
        Inputs: texts input strings; Outputs: list[list[float]] embeddings aligned with the input order.
        """
        return self._model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=False,
        )

def get_embedding_model() -> BGEEmbeddingModel:
    return BGEEmbeddingModel(model_name=settings.EMBEDDING_MODEL_NAME)
