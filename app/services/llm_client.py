import json
import re
import httpx
import time
from dataclasses import dataclass
from typing import Optional, Dict, Any

import openai


@dataclass
class QuestionExtractionResult:
    question: Optional[str]
    confidence: float
    original_text: Optional[str]
    position: Optional[int]
    
@dataclass
class AnswerExtractionResult:
    answer: Optional[str]
    relevance: float
    original_text: Optional[str]
    support_message_id: Optional[int]

@dataclass
class QualityMetrics:
    extraction_time: float
    token_count: int
    api_cost: float

class BaseAIClient:
    def __init__(self):
        self.quality_metrics = []
    
    def _build_optimized_question_prompt(self, dialog_text: str) -> str:
        """Enhanced English prompt for question extraction with JSON output"""
        prompt = f"""INSTRUCTION: Extract the first genuine support question from this dialog.

Return the response in the language used by the support agent and the user.

CRITERIA:
✓ Contains question words (how, what, where, when, why, can, is, does)
✓ Ends with "?" or implies seeking information/help
✓ Related to technical support, product features, or troubleshooting
✓ Excludes greetings, thanks, confirmations, and requests

ALGORITHM:
1. Read chronologically through USER messages only
2. Identify first message meeting criteria above
3. Normalize to clear, concise question (max 20 words)
4. Assign confidence score (0.0-1.0)

INPUT DIALOG:
{dialog_text}

OUTPUT FORMAT (JSON only):
{{
  "question": "normalized question text or null",
  "confidence": 0.95,
  "original_text": "exact user input",
  "position": "message number in dialog"
}}

EXAMPLES:
USER: "Hello! How do I reset my password?"
→ {{"question": "How do I reset my password?", "confidence": 0.95, "original_text": "Hello! How do I reset my password?", "position": 1}}

USER: "Please send me the instructions"  
→ {{"question": null, "confidence": 0.0, "original_text": "Please send me the instructions", "position": 1}}"""
        return prompt.strip()
    
    def _build_optimized_answer_prompt(self, question: str, dialog_text: str) -> str:
        """Enhanced English prompt for answer extraction with JSON output"""
        prompt = f"""INSTRUCTION: Find the direct answer to this specific question from SUPPORT responses.

Return the response in the language used by the support agent and the user.

TARGET QUESTION: {question}

CRITERIA:
✓ Response from SUPPORT role only
✓ Directly addresses the target question
✓ Contains actionable information or clear answer
✓ Excludes follow-up questions, requests for clarification
✓ Must appear after the question in dialog flow

ALGORITHM:
1. Locate target question in dialog
2. Find first SUPPORT response after question
3. Verify response directly answers the question
4. Extract core answer removing pleasantries
5. Assign relevance score

INPUT DIALOG:
{dialog_text}

OUTPUT FORMAT (JSON only):
{{
  "answer": "direct answer text or null",
  "relevance": 0.85,
  "original_text": "full support response",
  "support_message_id": "position in dialog"
}}

EXAMPLES:
Question: "How do I reset my password?"
SUPPORT: "To reset your password, go to Settings → Security → Reset Password"
→ {{"answer": "Go to Settings → Security → Reset Password", "relevance": 0.95, "original_text": "To reset your password, go to Settings → Security → Reset Password", "support_message_id": 2}}

Question: "Can I delete my account?"  
SUPPORT: "What specific issues are you having?"
→ {{"answer": null, "relevance": 0.0, "original_text": "What specific issues are you having?", "support_message_id": 2}}"""
        return prompt.strip()
    
    def _validate_qa_pair(self, question: str, answer: str) -> Dict[str, Any]:
        """Validate extracted Q&A pair quality"""
        validation_result = {
            "is_valid": True,
            "quality_score": 0.0,
            "issues": []
        }
        
        # Question validation
        if not question or len(question.strip()) < 5:
            validation_result["is_valid"] = False
            validation_result["issues"].append("Question too short or empty")
        
        if question and not any(word in question.lower() for word in ['how', 'what', 'where', 'when', 'why', 'can', 'is', 'does', '?']):
            validation_result["quality_score"] -= 0.3
            validation_result["issues"].append("Question lacks interrogative structure")
        
        # Answer validation
        if not answer or len(answer.strip()) < 3:
            validation_result["is_valid"] = False
            validation_result["issues"].append("Answer too short or empty")
            
        # Calculate quality score
        base_score = 1.0
        
        # Question clarity (30%)
        if question and '?' in question:
            base_score += 0.3
        elif question and any(word in question.lower() for word in ['how', 'what', 'where', 'when', 'why']):
            base_score += 0.2
            
        # Answer directness (40%)
        if answer and len(answer.split()) > 3:
            base_score += 0.3
        if answer and any(word in answer.lower() for word in ['go to', 'click', 'select', 'enter', 'set']):
            base_score += 0.1
            
        # Context relevance (20%)
        if question and answer:
            common_words = set(question.lower().split()) & set(answer.lower().split())
            if len(common_words) > 1:
                base_score += 0.2
                
        # Completeness (10%)
        if answer and len(answer) > 20:
            base_score += 0.1
            
        validation_result["quality_score"] = min(base_score, 1.0)
        
        if validation_result["quality_score"] < 0.5:
            validation_result["is_valid"] = False
            validation_result["issues"].append("Quality score too low")
            
        return validation_result

    def _remove_think_and_channels(self, text: str) -> str:
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
        
        if text.startswith("```"):
            parts = text.split("```")
            for part in parts:
                p = part.strip()
                if not p:
                    continue
                if p.lower().startswith(("json", "python", "text")):
                    lines = p.split('\n', 1)
                    if len(lines) > 1:
                        p = lines[1].strip()
                    else:
                        continue
                if p:
                    text = p
                    break
        
        text = text.strip()
        if text.startswith('"') and text.endswith('"'):
            text = text[1:-1].strip()
        
        return text

    def _extract_first_json_object(self, text: str) -> Optional[dict]:
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                return parsed
        except:
            pass
        
        import re
        json_pattern = r'\{[^{}]*\}'
        matches = re.findall(json_pattern, text)
        
        for match in matches:
            try:
                parsed = json.loads(match)
                if isinstance(parsed, dict):
                    return parsed
            except:
                continue
        
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
            else:
                if ch == '"':
                    in_string = True
                elif ch == '{':
                    if depth == 0:
                        start_index = index
                    depth += 1
                elif ch == '}':
                    if depth > 0:
                        depth -= 1
                        if depth == 0 and start_index != -1:
                            candidate = text[start_index:index + 1]
                            try:
                                parsed = json.loads(candidate)
                                if isinstance(parsed, dict):
                                    return parsed
                            except:
                                pass
        return None


class OpenAIClient(BaseAIClient):
    def __init__(
        self, 
        api_key: str, 
        model: str = "gpt-4o-mini",
        proxy_url: str = None,
        enable_monitoring: bool = False
    ):
        super().__init__()
        self.client = openai.OpenAI(api_key=api_key)
        if proxy_url:
            self.client = openai.OpenAI(
                api_key=api_key,
                http_client=httpx.Client(proxy=proxy_url)
            )
        self.model = model
        self.enable_monitoring = enable_monitoring
        self.default_config = {
            "temperature": 0.1,
            "top_p": 0.9,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
            "response_format": {"type": "json_object"}
        }

    def is_available(self) -> bool:
        try:
            self.client.models.list()
            return True
        except Exception:
            return False

    def extract_main_question(self, dialog_text: str, timeout: int = 30) -> Optional[QuestionExtractionResult]:
        """Extract question with confidence scoring and structured output"""
        start_time = time.time()
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert support dialog analyzer. Always respond with valid JSON."},
                    {"role": "user", "content": self._build_optimized_question_prompt(dialog_text)}
                ],
                **self.default_config,
                timeout=timeout
            )
            
            content = response.choices[0].message.content.strip()
            
            # Parse JSON response
            try:
                result_data = json.loads(content)
                result = QuestionExtractionResult(
                    question=result_data.get("question"),
                    confidence=float(result_data.get("confidence", 0.0)),
                    original_text=result_data.get("original_text"),
                    position=result_data.get("position")
                )
                
                # Record metrics
                if self.enable_monitoring:
                    processing_time = time.time() - start_time
                    self._record_metrics(processing_time, response.usage.total_tokens)
                
                return result
                
            except (json.JSONDecodeError, ValueError) as e:
                # Fallback to legacy parsing
                question = self._fallback_question_extraction(content)
                if question:
                    return QuestionExtractionResult(
                        question=question,
                        confidence=0.7,  # Lower confidence for fallback
                        original_text=content,
                        position=1
                    )
                
        except Exception as e:
            print(f"OpenAI API error: {e}")
            return None
                
        return None

    def extract_answer_for_question(self, question: str, dialog_text: str, timeout: int = 30) -> Optional[AnswerExtractionResult]:
        """Extract answer with relevance scoring and structured output"""
        start_time = time.time()
        
        if not self.is_available():
            return None
            
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert support response analyzer. Always respond with valid JSON."},
                    {"role": "user", "content": self._build_optimized_answer_prompt(question, dialog_text)}
                ],
                **self.default_config,
                max_tokens=200,
                timeout=timeout
            )
            
            content = response.choices[0].message.content.strip()
            
            # Parse JSON response
            try:
                result_data = json.loads(content)
                result = AnswerExtractionResult(
                    answer=result_data.get("answer"),
                    relevance=float(result_data.get("relevance", 0.0)),
                    original_text=result_data.get("original_text"),
                    support_message_id=result_data.get("support_message_id")
                )
                
                # Record metrics
                if self.enable_monitoring:
                    processing_time = time.time() - start_time
                    self._record_metrics(processing_time, response.usage.total_tokens)
                
                return result
                
            except (json.JSONDecodeError, ValueError) as e:
                # Fallback to legacy parsing
                answer = self._fallback_answer_extraction(content)
                if answer:
                    return AnswerExtractionResult(
                        answer=answer,
                        relevance=0.7,  # Lower relevance for fallback
                        original_text=content,
                        support_message_id=None
                    )
                    
        except Exception as e:
            print(f"OpenAI API error: {e}")
            return None
                
        return None
    
    def extract_qa_pair_with_validation(self, dialog_text: str) -> Optional[Dict[str, Any]]:
        """Extract and validate complete Q&A pair"""
        # Extract question
        question_result = self.extract_main_question(dialog_text)
        if not question_result or not question_result.question:
            return None
            
        # Skip low confidence questions
        if question_result.confidence < 0.5:
            return None
            
        # Extract answer
        answer_result = self.extract_answer_for_question(question_result.question, dialog_text)
        if not answer_result or not answer_result.answer:
            return None
            
        # Skip low relevance answers
        if answer_result.relevance < 0.5:
            return None
            
        # Validate Q&A pair
        validation = self._validate_qa_pair(question_result.question, answer_result.answer)
        if not validation["is_valid"]:
            return None
            
        return {
            "question": question_result.question,
            "answer": answer_result.answer,
            "question_confidence": question_result.confidence,
            "answer_relevance": answer_result.relevance,
            "quality_score": validation["quality_score"],
            "validation_issues": validation["issues"],
            "metadata": {
                "question_position": question_result.position,
                "answer_position": answer_result.support_message_id,
                "original_question_text": question_result.original_text,
                "original_answer_text": answer_result.original_text
            }
        }
    
    def _fallback_question_extraction(self, content: str) -> Optional[str]:
        """Fallback method for question extraction when JSON parsing fails"""
        cleaned = self._remove_think_and_channels(content)
        if not cleaned or cleaned == "NO_QUESTION":
            return None
            
        first_line = cleaned.splitlines()[0].strip()
        if first_line.startswith('"') and first_line.endswith('"'):
            first_line = first_line[1:-1].strip()
            
        if first_line in {"", "NO_QUESTION"}:
            return None
            
        return first_line
    
    def _fallback_answer_extraction(self, content: str) -> Optional[str]:
        """Fallback method for answer extraction when JSON parsing fails"""
        cleaned = self._remove_think_and_channels(content)
        if not cleaned or cleaned == "NO_ANSWER":
            return None
            
        first_line = cleaned.splitlines()[0].strip()
        if first_line in {"", "NO_ANSWER"}:
            return None
            
        if first_line.startswith('"') and first_line.endswith('"'):
            first_line = first_line[1:-1].strip()
            
        return first_line
    
    def _record_metrics(self, processing_time: float, token_count: int):
        """Record performance metrics"""
        # Cost calculation for gpt-4o-mini pricing
        if "gpt-4o-mini" in self.model:
            cost_per_1k_input_tokens = 0.00015  # $0.15 per 1M input tokens
            cost_per_1k_output_tokens = 0.0006   # $0.60 per 1M output tokens
            # Estimate 70% input, 30% output tokens
            estimated_cost = (token_count * 0.7 / 1000) * cost_per_1k_input_tokens + (token_count * 0.3 / 1000) * cost_per_1k_output_tokens
        elif "gpt-4" in self.model:
            cost_per_1k_tokens = 0.01  # GPT-4 pricing
            estimated_cost = (token_count / 1000) * cost_per_1k_tokens
        else:
            cost_per_1k_tokens = 0.002  # Default pricing
            estimated_cost = (token_count / 1000) * cost_per_1k_tokens
        
        metrics = QualityMetrics(
            extraction_time=processing_time,
            token_count=token_count,
            api_cost=estimated_cost
        )
        
        self.quality_metrics.append(metrics)
        
        # Keep only last 100 metrics to prevent memory issues
        if len(self.quality_metrics) > 100:
            self.quality_metrics = self.quality_metrics[-100:]
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        if not self.quality_metrics:
            return {}
            
        times = [m.extraction_time for m in self.quality_metrics]
        costs = [m.api_cost for m in self.quality_metrics]
        
        return {
            "total_extractions": len(self.quality_metrics),
            "avg_processing_time": sum(times) / len(times),
            "total_cost": sum(costs),
            "avg_cost_per_extraction": sum(costs) / len(costs)
        }
