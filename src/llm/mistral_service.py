"""LLM Service with Mistral AI integration (SDK v1)"""
import asyncio
import logging
import os
from typing import Optional
from mistralai import Mistral
from src.config import Config

class MistralLLMService:
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key or Config.LLM_API_KEY or os.getenv("MISTRAL_API_KEY")
        self.model = model or Config.LLM_MODEL or "mistral-large-latest"
        self.logger = logging.getLogger("mistral_llm")
        self.client = None
        if self.api_key:
            try:
                self.client = Mistral(api_key=self.api_key)  # v1 client
                self.logger.info("Mistral client initialized")
            except Exception as e:
                self.logger.error(f"Failed to initialize Mistral client: {e}")
        else:
            self.logger.warning("No Mistral API key provided; using mock responses")

    async def generate_response(self, prompt: str, system: str = "", temperature: float | None = None, max_tokens: int = 512) -> str:
        """Generic chat completion wrapper used by the RAG layer."""
        if not self.client:
            return await self._mock_response(prompt)

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        def _call():
            return self.client.chat.complete(
                model=self.model,
                messages=messages,
                temperature=Config.TEMPERATURE if temperature is None else temperature,
                max_tokens=max_tokens,
            )
        try:
            resp = await asyncio.to_thread(_call)
            return resp.choices[0].message.content
        except Exception as e:
            self.logger.error(f"Mistral API error: {e}")
            return await self._mock_response(prompt)

    async def _mock_response(self, prompt: str) -> str:
        await asyncio.sleep(0.1)
        return "(mock) I don't know based on the current document."
