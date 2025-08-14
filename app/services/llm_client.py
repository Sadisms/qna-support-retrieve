import json
import re

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
        except requests.RequestException:
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

    def _build_answer_prompt(self, question: str, dialog_text: str) -> str:
        prompt = f"""
        Ты помощник поддержки. Не выдумывай ответы и не формулируй новые.
        Твоя задача: по данному вопросу пользователя найти в диалоге первый явный ответ сотрудника поддержки.

        Правила:
        - Вопрос уже задан: используй ровно его формулировку ниже
        - Ответ должен быть взят ТОЛЬКО из реплик сотрудника поддержки
        - Если явного ответа нет, верни ровно NO_ANSWER
        - Верни ТОЛЬКО текст ответа без кавычек и пояснений

        Вопрос:
        <question>
        {question}
        </question>

        Диалог:
        <dialog>
        {dialog_text}
        </dialog>
        """
        return prompt.strip()

    def _remove_think_and_channels(self, text: str) -> str:
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
        if text.startswith("```"):
            parts = text.split("```")
            candidate = None
            for part in parts:
                p = part.strip()
                if not p:
                    continue
                if p.lower().startswith("json"):
                    p = p[4:].strip()
                if p:
                    candidate = p
                    break
            if candidate is not None:
                text = candidate

        text = text.strip()
        return text

    def _extract_first_json_object(self, text: str) -> Optional[dict]:
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass

        in_string = False
        is_escaped = False
        depth = 0
        start_index = -1
        for index, ch in enumerate(text):
            if in_string:
                if is_escaped:
                    is_escaped = False
                elif ch == "\\":
                    is_escaped = True
                elif ch == '"':
                    in_string = False
                continue
            else:
                if ch == '"':
                    in_string = True
                    continue
                if ch == '{':
                    if depth == 0:
                        start_index = index
                    depth += 1
                    continue
                if ch == '}':
                    if depth > 0:
                        depth -= 1
                        if depth == 0 and start_index != -1:
                            candidate = text[start_index:index + 1]
                            try:
                                parsed = json.loads(candidate)
                                if isinstance(parsed, dict):
                                    return parsed
                            except Exception:
                                pass
        return None

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
            cleaned = self._remove_think_and_channels(answer)
            if not cleaned:
                return None
            if cleaned.startswith("{") and cleaned.endswith("}"):
                parsed = self._extract_first_json_object(cleaned)
                if parsed and isinstance(parsed.get("question"), str):
                    q = parsed["question"].strip()
                    if q and q != "NO_QUESTION":
                        return q
                    return None
            first_line = cleaned.splitlines()[0].strip()
            if first_line.startswith('"') and first_line.endswith('"'):
                first_line = first_line[1:-1].strip()
            if first_line in {"", "NO_QUESTION"}:
                return None
            return first_line
        except requests.RequestException as e:
            return None

    def extract_answer_for_question(self, question: str, dialog_text: str, timeout: int = 60 * 10) -> Optional[str]:
        url = f"{self.base_url}/api/generate"

        payload = {
            "model": self.model,
            "prompt": self._build_answer_prompt(question, dialog_text),
            "max_tokens": 200,
            "temperature": 0.0,
            "top_p": 0.9,
            "stream": False,
        }

        if not self.is_available():
            return None

        try:
            response = requests.post(url, json=payload, timeout=timeout)
            response.raise_for_status()
            data = response.json()
            raw = data.get("response", "").strip()
            cleaned = self._remove_think_and_channels(raw)
            if not cleaned or cleaned == "NO_ANSWER":
                return None
            first_line = cleaned.splitlines()[0].strip()
            if first_line in {"", "NO_ANSWER"}:
                return None
            if first_line.startswith('"') and first_line.endswith('"'):
                first_line = first_line[1:-1].strip()
            return first_line
        except requests.RequestException:
            return None

    def extract_qa_pair(self, dialog_text: str, timeout: int = 60*10) -> Tuple[Optional[str], Optional[str]]:
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
            print("Model is not available")
            return None, None

        try:
            response = requests.post(url, json=payload, timeout=timeout)
            response.raise_for_status()
            data = response.json()
            raw = data.get("response", "").strip()
            cleaned = self._remove_think_and_channels(raw)
            print(cleaned)

            parsed = self._extract_first_json_object(cleaned)
            print(parsed)
            if parsed is None and ("Вопрос:" in cleaned or "Ответ:" in cleaned):
                question = None
                answer = None
                for line in cleaned.splitlines():
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

        except requests.RequestException as e:
            print(e)
            return None, None

        except Exception as e:
            print(e)
            return None, None
