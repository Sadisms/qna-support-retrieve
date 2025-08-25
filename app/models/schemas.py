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


class BaseResponse(BaseModel):
    status: Literal["success", "error"] = Field(..., description="Operation status")
    message: Optional[str] = Field(None, description="Result message")


class GetAnswerBody(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000, description="Question to search for")
    
    @validator('question')
    def validate_question(cls, v):
        if not v.strip():
            raise ValueError('Question cannot be empty')
        return v.strip()


class GetAnswerResponse(BaseModel):
    query: str = Field(..., description="Original query")
    answer: str = Field(..., description="Answer to the question")
    timestamp: datetime = Field(default_factory=datetime.now, description="Processing time")


class HealthCheckResponse(BaseModel):
    status: str = Field(..., description="Service status")
    timestamp: datetime = Field(default_factory=datetime.now, description="Check time")
    version: str = Field(default="0.1.0", description="Service version")


class AuthErrorResponse(BaseModel):
    detail: str = Field(..., description="Error message")
    timestamp: datetime = Field(default_factory=datetime.now, description="Error time")