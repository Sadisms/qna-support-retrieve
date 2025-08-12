import json

import requests

from typing import Optional, Tuple


class OllamaClient:
    def __init__(self, base_url: str, model: str):
        self.base_url = base_url.rstrip('/')
        self.model = model

    def _build_question_prompt(self, dialog_text: str) -> str:
        prompt = f"""
        Ты помощник поддержки.

        Твоя задача: из диалога между пользователем и сотрудником
        поддержки найти первый осмысленный вопрос пользователя.

        Правила:
        - Учитывай только реплики пользователя.
        - Игнорируй приветствия, благодарности, оффтоп,
          риторические вопросы и уточнения типа
          "правильно ли я понял".
        - Игнорируй вопросы и подсказки сотрудника поддержки.
        - Если в одной реплике несколько вопросов, выбери самый первый.
        - Переформулируй в нейтральной форме так, чтобы вопрос был
          понятен вне контекста диалога.
        - Убери обращения и лишние детали.

        Формат ответа:
        - Только один вопрос, одна короткая фраза.
        - Без префиксов и суффиксов.
        - Без кавычек и нумерации.
        - Заверши вопрос знаком вопроса.
        - Если явного вопроса пользователя нет, ответь ровно: NO_QUESTION

        Диалог:
        <dialog>
        {dialog_text}
        </dialog>
        """
        return prompt.strip()

    def _build_qa_prompt(self, dialog_text: str) -> str:
        prompt = f"""
        Ты помощник поддержки.

        Твоя задача: из диалога между пользователем и сотрудником
        поддержки найти первую пару Вопрос-Ответ.

        Правила извлечения:
        - Вопрос берётся только из реплик пользователя.
        - Игнорируй приветствия, благодарности, оффтоп,
          риторические вопросы и уточнения типа
          "правильно ли я понял".
        - Если в одной реплике несколько вопросов, выбери самый первый.
        - Ответ берётся из ближайшей релевантной реплики сотрудника поддержки,
          которая непосредственно отвечает на этот вопрос.
        - Переформулируй вопрос в нейтральной форме, понятной вне контекста.
        - Ответ сделай кратким и конкретным, без лишних деталей.

        Формат ответа:
        Верни ТОЛЬКО JSON-объект без пояснений и форматирования кода:
        {"question": "...", "answer": "..."}
        - "question": первый осмысленный вопрос пользователя одной короткой фразой,
          с вопросительным знаком в конце; если нет, укажи NO_QUESTION.
        - "answer": краткий ответ саппорта на этот вопрос; если нет, укажи NO_ANSWER.

        Диалог:
        <dialog>
        {dialog_text}
        </dialog>
        """
        return prompt.strip()

    def _build_prompt(self, dialog_text: str) -> str:
        return self._build_question_prompt(dialog_text)

    def extract_main_question(self, dialog_text: str, timeout: int = 300) -> Optional[str]:
        url = f"{self.base_url}/api/generate"

        payload = {
            "model": self.model,
            "prompt": self._build_question_prompt(dialog_text),
            "max_tokens": 100,
            "temperature": 0.2,
            "top_p": 0.95,
            "stream": False
        }

        try:
            response = requests.post(url, json=payload, timeout=timeout)
            response.raise_for_status()
            data = response.json()

            answer = data.get("response", "").strip()
            if answer == "":
                return None
            return answer
        except requests.RequestException as e:
            return None

    def extract_qa_pair(self, dialog_text: str, timeout: int = 20) -> Tuple[Optional[str], Optional[str]]:
        url = f"{self.base_url}/api/generate"

        payload = {
            "model": self.model,
            "prompt": self._build_qa_prompt(dialog_text),
            "max_tokens": 200,
            "temperature": 0.2,
            "top_p": 0.95,
            "stream": False,
        }

        try:
            response = requests.post(url, json=payload, timeout=timeout)
            response.raise_for_status()
            data = response.json()
            raw = data.get("response", "").strip()

            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                parsed = None
                if "```" in raw:
                    parts = raw.split("```")
                    for part in parts:
                        part_stripped = part.strip()
                        if part_stripped.startswith("json"):
                            part_stripped = part_stripped[4:].strip()
                        try:
                            parsed = json.loads(part_stripped)
                            break
                        except Exception:
                            continue
                if parsed is None and ("Вопрос:" in raw or "Ответ:" in raw):
                    question = None
                    answer = None
                    for line in raw.splitlines():
                        line_stripped = line.strip()
                        if line_stripped.lower().startswith("вопрос:") and question is None:
                            question = line_stripped.split(":", 1)[1].strip()
                        if line_stripped.lower().startswith("ответ:") and answer is None:
                            answer = line_stripped.split(":", 1)[1].strip()
                    return question or None, answer or None

            if isinstance(parsed, dict):
                question = parsed.get("question")
                answer = parsed.get("answer")
                question = question.strip() if isinstance(question, str) else None
                answer = answer.strip() if isinstance(answer, str) else None
                if question in {"", "NO_QUESTION"}:
                    question = None
                if answer in {"", "NO_ANSWER"}:
                    answer = None
                return question, answer

            return None, None
        except requests.RequestException:
            return None, None
