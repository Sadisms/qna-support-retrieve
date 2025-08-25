from typing import Any

import httpx

import numpy as np
import openai
from numpy import ndarray, dtype
from app.core.exceptions import EmbeddingException, LLMException
import json

SYSTEM_PROMPT = """You are a high-precision Retrieval-Augmented Generation assistant.
Strict output policy:
- Use ONLY facts from <context>. If not fully supported, respond with a structured refusal.
- Output must be a single JSON object in the user's language (Russian if the user writes in Russian) with exactly one of the following forms and no extra keys, no prose before/after:
  {"answer": "..."}
  {"no_answer": true}
- The answer must be concise (1–5 sentences), contain no questions, no requests for clarification, no disclaimers, no metadata.
- If items conflict and cannot be resolved from <context>, return {"no_answer": true}.
"""

PROMT = """
<context>
{context}
</context>

<query>
{query}
</query>
"""

class OpenAIClient:
    EMBEDDING_MODEL = "text-embedding-3-small"
    EMBEDDING_FORMAT = "float"

    def __init__(self, api_key: str, model: str, proxy_url: str):
        self.client = openai.OpenAI(api_key=api_key)
        if proxy_url:
            self.client = openai.OpenAI(
                api_key=api_key,
                http_client=httpx.Client(proxy=proxy_url)
            )
        self.model = model


    def embedding(self, text: str) -> ndarray:
        try:
            response = self.client.embeddings.create(
                model=self.EMBEDDING_MODEL,
                input=text,
                encoding_format=self.EMBEDDING_FORMAT
            )
            return np.array(response.data[0].embedding)
        except Exception as exc:
            raise EmbeddingException("Embedding request failed", {"reason": str(exc)})
    
    def rag_search(self, query: str, context: list[dict]) -> str:
        """
            context is a list of dicts with the following keys:
            - question: str
            - dialog: list[dict]
            - score: float
            - ticket_id: int
        """

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=0,
                top_p=1,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": PROMT.format(
                            query=query,
                            context=str(context)
                        )
                    }
                ]
            )
            content = (response.choices[0].message.content or "").strip()
        except Exception as exc:
            raise LLMException("LLM generation failed", {"reason": str(exc)})

        # Parse and validate strict JSON response
        try:
            payload = json.loads(content)
        except Exception:
            raise LLMException("LLM returned non-JSON response", {"raw": content[:200]})

        if isinstance(payload, dict) and payload.get("no_answer") is True:
            raise LLMException("No answer found", {"query": query})

        answer = payload.get("answer") if isinstance(payload, dict) else None
        if not isinstance(answer, str) or not answer.strip():
            raise LLMException("Invalid answer payload", {"payload": payload})

        normalized = answer.strip()
        lowered = normalized.lower()
        # Heuristics to catch indirect refusals or clarification requests
        disallowed_markers = [
            "уточните", "что именно", "нет информации", "не найдено", "не нашёл", "не нашел",
            "i don't know", "do not have enough", "no information", "please clarify", "cannot answer",
        ]
        if any(marker in lowered for marker in disallowed_markers):
            raise LLMException("No grounded answer in context", {"answer": normalized})
        if "?" in normalized:
            raise LLMException("Answer must not contain questions", {"answer": normalized})

        return normalized
