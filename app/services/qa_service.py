from typing import List

from sqlalchemy.orm import Session as SQLAlchemySession

from app.models.database import QAModel


def save_qa(
    db: SQLAlchemySession, 
    ticket_id: int, 
    question: str, 
    answer: str, 
    source: dict
) -> int:
    qa = QAModel(
        ticket_id=ticket_id,
        question=question,
        answer=answer,
        source=source
    )
    db.add(qa)
    db.commit()
    db.refresh(qa)
    return qa.id


def get_qa(db: SQLAlchemySession, ticket_ids: List[int]) -> List[QAModel]:
    return db.query(QAModel).filter(QAModel.ticket_id.in_(ticket_ids)).all()
