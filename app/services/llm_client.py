from typing import Any

import httpx

import numpy as np
import openai
from numpy import ndarray, dtype
from app.core.exceptions import EmbeddingException, LLMException

SYSTEM_PROMPT = """You are a high-precision Retrieval-Augmented Generation assistant.
Strict output policy:
- Use ONLY facts from <context>. If not fully supported, output exactly NO_ANSWER.
- Do not hallucinate or invent anything not present in <context>.
- Output must be ONE of:
  1) A single direct answer with no preface, no metadata, no disclaimers.
  2) Exactly: NO_ANSWER
- Match the user's language (Russian if user writes in Russian).
- Be concise (1–5 sentences). No follow-up questions. No formatting beyond plain text.
- If items conflict, state the conflict briefly; if unresolved, output NO_ANSWER.
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
                temperature=0.2,
                top_p=0.9,
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

        if not content or content.upper() == "NO_ANSWER":
            raise LLMException("No answer found", {"query": query})

        return content
