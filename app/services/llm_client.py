import httpx

import openai

SYSTEM_PROMPT = """You are a high-precision Retrieval-Augmented Generation assistant.
Follow these rules strictly:
- Use ONLY facts from <context>. If the answer isn't fully supported, say so and ask one clarifying question.
- Do not hallucinate. Do not invent data, IDs, URLs, or steps not present in <context>.
- Prefer the most relevant items; ignore obviously irrelevant/low-score noise.
- If you rely on a ticket, cite it as [ticket:{ticket_id}] from the item where it appears.
- If the user writes in Russian, answer in Russian; otherwise, match the user's language.
- Be concise (3–6 sentences). Use bullet points when helpful. No generic disclaimers.
- If items conflict, note the conflict and prefer the most recent dialog entry.
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


    def embedding(self, text: str) -> list[float]:
        response = openai.embeddings.create(
            model=self.EMBEDDING_MODEL,
            input=text,
            encoding_format=self.EMBEDDING_FORMAT
        )
        return response.data[0].embedding
    
    def rag_search(self, query: str, context: list[dict]) -> str:
        """
            context is a list of dicts with the following keys:
            - question: str
            - dialog: list[dict]
            - score: float
            - ticket_id: int
        """

        response = openai.chat.completions.create(
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
        return response.choices[0].message.content
