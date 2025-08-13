from datetime import datetime

from sqlalchemy import Column, Integer, Text, DateTime, JSON
from sqlalchemy.orm import declarative_base

BaseModel = declarative_base()


class QAModel(BaseModel):
    __tablename__ = "qa"
    
    id = Column(Integer, primary_key=True, index=True)
    ticket_id = Column(Integer, nullable=False)
    question = Column(Text, nullable=False)
    answer = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.now)
    source = Column(JSON, nullable=True)
