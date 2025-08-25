from typing import Any, Coroutine

from fastapi import APIRouter, HTTPException, Depends

from app.core.exceptions import LLMException, EmbeddingException, VectorStoreException
from app.core.auth import get_current_workspace
from app.models.schemas import SaveQABody, BaseResponse, GetAnswerBody, GetAnswerResponse, RoleType
from app.services.llm_client import OpenAIClient
from app.services.qdrant import QdrantHelper
from app.core.config import get_config


router = APIRouter(prefix="/qa", tags=["QA Operations"])


config = get_config()


llm_client = OpenAIClient(
    api_key=config.openai_api_key,
    model=config.openai_model,
    proxy_url=config.openai_proxy_url,
)


def get_qdrant_helper(workspace_id: str) -> QdrantHelper:
    return QdrantHelper(
        url=config.qdrant_url,
        collection_name=config.qdrant_collection_name,
        workspace_id=workspace_id
    )


@router.post("/save", response_model=BaseResponse)
async def save_qa_handler(body: SaveQABody, workspace_id: str = Depends(get_current_workspace)) -> BaseResponse:
    try:
        full_dialog_text = "\n".join([
            (("USER" if msg.role == RoleType.USER else "SUPPORT") + ": " + msg.content)
            for msg in body.dialog
        ])

        embedding = llm_client.embedding(full_dialog_text)
        qdrant_helper = get_qdrant_helper(workspace_id)
        qdrant_helper.add_vector(embedding, body.model_dump())

        return BaseResponse(
            status="success",
            message="Ticket successfully saved"
        )

    except (LLMException, EmbeddingException, VectorStoreException):
        raise

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.post("/search", response_model=GetAnswerResponse)
async def get_answer_handler(body: GetAnswerBody, workspace_id: str = Depends(get_current_workspace)) -> BaseResponse | GetAnswerResponse:
    try:
        embeding = llm_client.embedding(body.question)
        qdrant_helper = get_qdrant_helper(workspace_id)
        results = qdrant_helper.search_similar(embeding, top_k=5)
        if not results:
            raise VectorStoreException("No results found", {"query": body.question})

        llm_answer = llm_client.rag_search(body.question, results)

        return GetAnswerResponse(
            query=body.question,
            answer=llm_answer
        )
        
    except (EmbeddingException, VectorStoreException):
        raise

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

