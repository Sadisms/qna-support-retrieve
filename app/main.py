from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager

from app.core.config import get_config
from app.core.database import engine
from app.core.exceptions import QnAException
from app.core.error_handlers import (
    qna_exception_handler,
    http_exception_handler,
    general_exception_handler
)
from app.models.database import BaseModel
from app.api.qa_routes import router as qa_router
from app.api.health_routes import router as health_router


@asynccontextmanager
async def lifespan(_: FastAPI):
    BaseModel.metadata.create_all(bind=engine)
    yield


config = get_config()

app = FastAPI(
    title="QnA-Support-Retriever",
    lifespan=lifespan
)

app.add_exception_handler(QnAException, qna_exception_handler)
app.add_exception_handler(HTTPException, http_exception_handler)
app.add_exception_handler(Exception, general_exception_handler)

app.include_router(health_router)
app.include_router(qa_router)
