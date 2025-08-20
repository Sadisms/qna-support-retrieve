from datetime import datetime

from sqlalchemy import Column, Integer, Text, DateTime, JSON, String, Index
from sqlalchemy.orm import declarative_base

BaseModel = declarative_base()


class QAModel(BaseModel):
    __tablename__ = "qa"
    
    id = Column(Integer, primary_key=True, index=True)
    workspace_id = Column(String(50), nullable=False, index=True)
    ticket_id = Column(Integer, nullable=False)
    question = Column(Text, nullable=False)
    answer = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.now)
    source = Column(JSON, nullable=True)
    
    __table_args__ = (
        Index('ix_workspace_ticket', 'workspace_id', 'ticket_id'),
    )
