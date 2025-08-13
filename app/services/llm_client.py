import json

import requests

from typing import Optional, Tuple


class OllamaClient:
    def __init__(self, base_url: str, model: str):
        self.base_url = base_url.rstrip('/')
        self.model = model

    def is_available(self, timeout: int = 5) -> bool:
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=timeout)
            return response.status_code == 200
        except:
            return False

    def _build_question_prompt(self, dialog_text: str) -> str:
        prompt = f"""
        Ты помощник поддержки. Ты НИКОГДА не отвечаешь на вопросы и не даёшь советов.
        Твоя единственная задача: из диалога выделить первый явный вопрос пользователя.

        Что считать вопросом пользователя:
        - Реплика пользователя в вопросительной форме (обычно со знаком вопроса)
        - или начинается с вопросительных слов: как, когда, где, зачем, почему, какой/какая/какие,
          сколько, можно ли, что, кто.

        Что НЕ считать вопросом (верни NO_QUESTION):
        - Просьбы/команды: «сделайте», «нужно», «скиньте», «пришлите», «помогите», «оформите» и т.п.
        - Приветствия, благодарности, оффтоп
        - Риторические и уточняющие «правильно ли я понял», «верно?» и т.п.
        - Вопросы и подсказки сотрудника поддержки

        Правила извлечения:
        - Учитывай ТОЛЬКО реплики пользователя
        - Если в одной реплике несколько вопросов, выбери самый первый
        - Переформулируй кратко и нейтрально, понятно вне контекста
        - Убери обращения и лишние детали

        Формат ответа (строго):
        - Если вопрос найден: верни ТОЛЬКО одну короткую фразу-вопрос без кавычек, заканчивающуюся «?»
        - Если вопроса нет: верни ровно NO_QUESTION
        - Никаких префиксов/пояснений/кода

        Примеры:
        <example>
        USER: Привет!
        USER: Как сменить пароль?
        SUPPORT: Откройте настройки…
        => Как сменить пароль?
        </example>

        <example>
        USER: Пожалуйста, пришлите инструкцию по восстановлению
        SUPPORT: Могу помочь
        => NO_QUESTION
        </example>

        <example>
        USER: Мне не приходит код, это из-за тарифа?
        USER: Тогда перезвоните
        SUPPORT: Уточните номер
        => Мне не приходит код, это из-за тарифа?
        </example>

        Диалог:
        <dialog>
        {dialog_text}
        </dialog>
        """
        return prompt.strip()

    def _build_qa_prompt(self, dialog_text: str) -> str:
        prompt = f"""
        Ты помощник поддержки. Не выдумывай ответы и не формулируй новые.
        Задача: из диалога найти первую пару Вопрос-Ответ.

        Определи первый ВОПРОС пользователя по тем же критериям, что и в задаче выше.
        Если вопроса нет, верни question=NO_QUESTION и answer=NO_ANSWER.

        Правила:
        - Вопрос берётся ТОЛЬКО из реплик пользователя
        - Если в одной реплике несколько вопросов, выбери самый первый
        - Переформулируй вопрос кратко и нейтрально, с «?» в конце
        - Ответ возьми из ближайшей реплики сотрудника, которая прямо отвечает на этот вопрос
        - Если явного ответа нет, поставь NO_ANSWER
        - Никаких пояснений, только JSON

        Формат ответа (строго, одна строка, валидный JSON):
        {{"question": "...", "answer": "..."}}

        Примеры:
        <example>
        USER: Как сменить тариф?
        SUPPORT: Перейдите в раздел Тарифы…
        => {{"question": "Как сменить тариф?", "answer": "Перейдите в раздел Тарифы…"}}
        </example>

        <example>
        USER: Пришлите инструкцию по возврату
        SUPPORT: Готов помочь
        => {{"question": "NO_QUESTION", "answer": "NO_ANSWER"}}
        </example>

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
            "temperature": 0.0,
            "top_p": 0.9,
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

    def extract_qa_pair(self, dialog_text: str, timeout: int = 60) -> Tuple[Optional[str], Optional[str]]:
        url = f"{self.base_url}/api/generate"

        payload = {
            "model": self.model,
            "prompt": self._build_qa_prompt(dialog_text),
            "max_tokens": 180,
            "temperature": 0.0,
            "top_p": 0.9,
            "stream": False,
        }

        if not self.is_available():
            return None, None

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

        except Exception:
            return None, None
