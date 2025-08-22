import json
import re
import httpx

import requests
import openai

from typing import Optional


class BaseAIClient:
    def _build_question_prompt(self, dialog_text: str) -> str:
        prompt = f"""
        Ты помощник поддержки. Ты НИКОГДА не отвечаешь на вопросы, не даёшь советы и не генерируешь новый контент.
        Твоя единственная задача: из предоставленного диалога выделить первый явный вопрос пользователя, строго следуя правилам ниже.
        Если правила не позволяют выделить вопрос, верни ровно NO_QUESTION.

        Определение вопроса пользователя:
        - Это реплика пользователя в явно вопросительной форме, обычно заканчивающаяся знаком вопроса (?).
        - Или реплика, начинающаяся с вопросительных слов: как, когда, где, зачем, почему, какой/какая/какие/какое, сколько, можно ли, что, кто, есть ли, ли.
        - Вопрос должен быть прямым и требовать информации или уточнения.

        Что НЕ считать вопросом (обязательно верни NO_QUESTION):
        - Любые просьбы, команды или инструкции: «сделайте», «нужно», «скиньте», «пришлите», «помогите», «оформите», «отправьте», «закажите», «измените» и подобные.
        - Приветствия: «привет», «здравствуйте», «добрый день» и т.п.
        - Благодарности: «спасибо», «благодарю» и т.п.
        - Оффтоп, жалобы или утверждения без вопросительной формы: «не работает», «проблема с...», «я хочу...».
        - Риторические вопросы: «почему так сложно?», «зачем это нужно?» если они не требуют реального ответа.
        - Уточняющие фразы: «правильно ли я понял», «верно?», «да?» если они не являются полноценным вопросом.
        - Любые реплики сотрудника поддержки, даже если они содержат вопросы или подсказки.
        - Если реплика пользователя смешана (вопрос + просьба), игнорируй, если она не начинается как чистый вопрос.

        Строгие правила извлечения:
        - Учитывай ТОЛЬКО реплики, помеченные как USER:. Игнорируй все SUPPORT: полностью.
        - Просканируй диалог сверху вниз: найди САМУЮ ПЕРВУЮ реплику пользователя, которая соответствует определению вопроса.
        - Если в одной реплике несколько вопросов, извлеки только самый первый из них.
        - Переформулируй извлеченный вопрос кратко, нейтрально и понятно вне контекста диалога.
        - Убери все обращения (типа «скажите пожалуйста»), лишние детали, эмоции и ненужные слова. Сделай фразу чистым вопросом.
        - Вопрос должен заканчиваться знаком «?» и быть одной короткой фразой (максимум 15-20 слов).
        - Не добавляй ничего от себя: не исправляй, не уточняй, не придумывай.
        - Если после сканирования всего диалога вопроса нет, верни ровно NO_QUESTION.

        Формат ответа (строго соблюдай, никаких исключений):
        - Если вопрос найден: верни ТОЛЬКО одну короткую фразу-вопрос без кавычек, заканчивающуюся «?».
        - Если вопроса нет: верни ровно NO_QUESTION.
        - Никаких префиксов, пояснений, дополнительного текста, кода или JSON. Только указанный вывод.

        Примеры (изучай внимательно):
        <example>
        USER: Привет!
        USER: Как сменить пароль?
        SUPPORT: Откройте настройки…
        => Как сменить пароль?
        </example>

        <example>
        USER: Пожалуйста, пришлите инструкцию по восстановлению.
        SUPPORT: Могу помочь.
        => NO_QUESTION
        </example>

        <example>
        USER: Мне не приходит код, это из-за тарифа?
        USER: Тогда перезвоните.
        SUPPORT: Уточните номер.
        => Почему не приходит код?
        </example>

        <example>
        USER: Сделайте возврат денег.
        USER: Почему так долго?
        SUPPORT: Обработка занимает 3 дня.
        => Почему так долго?
        </example>

        <example>
        USER: Здравствуйте, помогите пожалуйста с заказом.
        SUPPORT: Что именно?
        USER: Верно ли, что доставка бесплатная?
        => Верно ли, что доставка бесплатная?
        </example>

        <example>
        USER: Спасибо за помощь!
        USER: Всё работает.
        => NO_QUESTION
        </example>

        <example>
        SUPPORT: Есть вопросы?
        USER: Нет, спасибо.
        => NO_QUESTION
        </example>

        <example>
        USER: Можно ли отменить подписку? И как это сделать?
        => Можно ли отменить подписку?
        </example>

        Диалог для анализа:
        <dialog>
        {dialog_text}
        </dialog>
        """
        return prompt.strip()

    def _build_answer_prompt(self, question: str, dialog_text: str) -> str:
        prompt = f"""
        Ты помощник поддержки. Ты НИКОГДА не выдумываешь ответы, не формулируешь новые и не генерируешь контент.
        Твоя единственная задача: по данному точному вопросу пользователя найти в диалоге первый явный ответ сотрудника поддержки, строго следуя правилам.
        Если правила не позволяют найти ответ, верни ровно NO_ANSWER.

        Определение ответа сотрудника:
        - Это реплика SUPPORT:, которая напрямую адресует предоставленный вопрос.
        - Ответ должен быть полным и взятым verbatim (слово в слово) из реплики SUPPORT:.
        - Это может быть объяснение, инструкция или информация, релевантная вопросу.

        Что НЕ считать ответом (обязательно верни NO_ANSWER):
        - Реплики пользователя (USER:), даже если они содержат информацию.
        - Вопросы или уточнения от SUPPORT: (например, «Уточните...», «Что именно?»).
        - Ответы, не относящиеся напрямую к данному вопросу.
        - Если ответ неявный или частичный, игнорируй.
        - Если в диалоге нет реплики SUPPORT: после вопроса или она не отвечает на него.

        Строгие правила извлечения:
        - Используй ровно предоставленную формулировку вопроса ниже — не меняй её.
        - Просканируй диалог сверху вниз: найди САМУЮ ПЕРВУЮ реплику SUPPORT:, которая даёт прямой ответ на вопрос.
        - Если в одной реплике SUPPORT: несколько частей, возьми только релевантную часть, но без изменений текста.
        - Не добавляй, не удаляй и не исправляй текст ответа.
        - Ответ может быть многострочным, но убери лишние обращения или нерелевантные части, если они не часть ответа.
        - Если после сканирования всего диалога ответа нет, верни ровно NO_ANSWER.

        Формат ответа (строго соблюдай, никаких исключений):
        - Если ответ найден: верни ТОЛЬКО текст ответа без кавычек, префиксов или пояснений.
        - Если ответа нет: верни ровно NO_ANSWER.
        - Никаких дополнительных текстов, JSON или кода. Только указанный вывод.

        Примеры (изучай внимательно):
        <example>
        Вопрос: Как сменить пароль?
        Диалог:
        USER: Как сменить пароль?
        SUPPORT: Откройте настройки и выберите "Сменить пароль".
        => Откройте настройки и выберите "Сменить пароль".
        </example>

        <example>
        Вопрос: Почему не приходит код?
        Диалог:
        USER: Мне не приходит код.
        SUPPORT: Уточните номер.
        => NO_ANSWER
        </example>

        <example>
        Вопрос: Верно ли, что доставка бесплатная?
        Диалог:
        USER: Верно ли, что доставка бесплатная?
        SUPPORT: Да, для заказов от 1000 руб.
        USER: Спасибо.
        => Да, для заказов от 1000 руб.
        </example>

        <example>
        Вопрос: Можно ли отменить подписку?
        Диалог:
        USER: Можно ли отменить подписку?
        SUPPORT: Что с подпиской?
        SUPPORT: Да, можно через личный кабинет.
        => Да, можно через личный кабинет.
        </example>

        <example>
        Вопрос: Как оформить возврат?
        Диалог:
        USER: Как оформить возврат?
        SUPPORT: Заполните форму на сайте.
        => Заполните форму на сайте.
        </example>

        Вопрос для анализа:
        <question>
        {question}
        </question>

        Диалог для анализа:
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


class OllamaClient(BaseAIClient):
    def __init__(self, base_url: str, model: str):
        self.base_url = base_url.rstrip('/')
        self.model = model

    def is_available(self, timeout: int = 5) -> bool:
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=timeout)
            return response.status_code == 200
        except requests.RequestException:
            return False

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
            print(data)
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
            print(data)
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


class OpenAIClient(BaseAIClient):
    def __init__(
        self, 
        api_key: str, 
        model: str = "gpt-4.1",
        proxy_url: str = None
    ):
        self.client = openai.OpenAI(api_key=api_key)
        if proxy_url:
            self.client = openai.OpenAI(
                api_key=api_key,
                http_client=httpx.Client(proxy=proxy_url)
            )
        self.model = model

    def is_available(self) -> bool:
        try:
            self.client.models.list()
            return True
        except Exception:
            return False

    def extract_main_question(self, dialog_text: str, timeout: int = 300) -> Optional[str]:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": self._build_question_prompt(dialog_text)}
                ],
                max_tokens=100,
                temperature=0.0,
                timeout=timeout
            )
            
            answer = response.choices[0].message.content.strip()
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
        except Exception as e:
            print(f"OpenAI API error: {e}")
            return None

    def extract_answer_for_question(self, question: str, dialog_text: str, timeout: int = 60 * 10) -> Optional[str]:
        if not self.is_available():
            return None

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": self._build_answer_prompt(question, dialog_text)}
                ],
                max_tokens=200,
                temperature=0.0,
                timeout=timeout
            )
            
            raw = response.choices[0].message.content.strip()
            cleaned = self._remove_think_and_channels(raw)
            
            if not cleaned or cleaned == "NO_ANSWER":
                return None
            
            first_line = cleaned.splitlines()[0].strip()
            if first_line in {"", "NO_ANSWER"}:
                return None
            
            if first_line.startswith('"') and first_line.endswith('"'):
                first_line = first_line[1:-1].strip()
            
            return first_line
        except Exception as e:
            print(f"OpenAI API error: {e}")
            return None
