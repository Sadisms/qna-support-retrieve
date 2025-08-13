from typing import List, Literal, Optional
from enum import Enum
from datetime import datetime

from pydantic import BaseModel, Field, validator


class RoleType(str, Enum):
    USER = "user"
    SUPPORT = "support"


class Dialog(BaseModel):
    role: RoleType
    content: str

    def is_user(self) -> bool:
        return self.role == RoleType.USER
    
    def is_support(self) -> bool:
        return self.role == RoleType.SUPPORT


class SaveQABody(BaseModel):
    ticket_id: int
    question: str
    dialog: List[Dialog]


class SaneQAResponse(BaseModel):
    status: Literal["success", "error"] = Field(..., description="Operation status")
    message: Optional[str] = Field(None, description="Result message")
    extracted_question: Optional[str] = Field(None, description="Extracted question")
    extracted_answer: Optional[str] = Field(None, description="Extracted answer")
    ticket_id: Optional[int] = Field(None, description="Ticket ID")
    already_saved: bool = Field(default=False, description="Indicator that the ticket has already been saved")
    timestamp: datetime = Field(default_factory=datetime.now, description="Processing time")


class GetAnswerBody(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000, description="Question to search for")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of results")
    
    @validator('question')
    def validate_question(cls, v):
        if not v.strip():
            raise ValueError('Question cannot be empty')
        return v.strip()


class GetAnswerResultResponse(BaseModel):
    question: str = Field(..., description="Found question")
    answer: str = Field(..., description="Corresponding answer")
    similarity: float = Field(..., ge=0.0, le=1.0, description="Similarity score")
    ticket_id: int = Field(..., description="Ticket ID")
    
    class Config:
        validate_assignment = True


class GetAnswerResponse(BaseModel):
    query: str = Field(..., description="Original query")
    results: List[GetAnswerResultResponse] = Field(..., description="Search results")
    total_found: int = Field(..., ge=0, description="Total number of found results")
    timestamp: datetime = Field(default_factory=datetime.now, description="Processing time")


class HealthCheckResponse(BaseModel):
    status: str = Field(..., description="Service status")
    timestamp: datetime = Field(default_factory=datetime.now, description="Check time")
    version: str = Field(default="0.1.0", description="Service version")


class AuthErrorResponse(BaseModel):
    detail: str = Field(..., description="Error message")
    timestamp: datetime = Field(default_factory=datetime.now, description="Error time")