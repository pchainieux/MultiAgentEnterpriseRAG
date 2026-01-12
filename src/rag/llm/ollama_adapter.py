from __future__ import annotations

import os
from typing import List, Optional

import requests
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.outputs import Generation, LLMResult

from src.app.core.config import settings


class OllamaChatModel(BaseChatModel):
    """
    LangChain compatible wrapper around Ollama's /api/generate endpoint. Configuration is read from settings / env to keep this Pydantic friendly.
    """
    temperature: float = 0.1  

    @property
    def _llm_type(self) -> str:
        return "ollama-chat"

    @property
    def _base_url(self) -> str:
        return settings.OLLAMA_BASE_URL.rstrip("/")

    @property
    def _model_name(self) -> str:
        return settings.OLLAMA_MODEL_NAME

    @property
    def _api_key(self) -> str:
        return settings.OLLAMA_API_KEY

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager=None,
        **kwargs,
    ) -> LLMResult:
        """
        Convert LangChain chat messages into a plain text prompt and call the Ollama compatible /generate endpoint to obtain a completion.
        Inputs: messages, stop, kwargs (such as temperature); Outputs: an LLMResult containing a single generated assistant text.
        """
        lines: List[str] = []
        for m in messages:
            role = m.type.upper()
            lines.append(f"{role}: {m.content}")
        prompt = "\n\n".join(lines)

        url = f"{self._base_url}/generate"
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self._model_name,
            "prompt": prompt,
            "stream": False,
        }

        resp = requests.post(url, headers=headers, json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()

        content = data.get("response", "") or ""

        gen = Generation(text=content)
        return LLMResult(generations=[[gen]])

    def invoke(self, input, **kwargs):
        """
        Provide a simple invoke() convenience wrapper that returns an object exposing a 'content' attribute for compatibility with the rest of the codebase.
        Inputs: input and optional kwargs; Outputs: a lightweight response object with a 'content' string field.
        """
        result = self._generate(input, **kwargs)
        return type("Resp", (), {"content": result.generations[0][0].text})


def get_ollama_llm() -> OllamaChatModel:
    """
    Construct an OllamaChatModel instance used by the provider router when LLM_PROVIDER=ollama.
    Inputs: none ; Outputs: an OllamaChatModel instance.
    """
    return OllamaChatModel()
