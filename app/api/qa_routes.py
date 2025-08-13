from fastapi import APIRouter, HTTPException, Depends

from app.core.database import get_db_context
from app.core.exceptions import DatabaseException, LLMException, EmbeddingException, VectorStoreException
from app.core.auth import get_current_user
from app.models.schemas import SaveQABody, SaneQAResponse, GetAnswerBody, GetAnswerResponse, GetAnswerResultResponse, RoleType
from app.services.qa_service import save_qa, get_qa
from app.services.llm_client import OllamaClient
from app.services.embeddings import Embedder
from app.services.qdrant import QdrantHelper
from app.core.config import get_config

router = APIRouter(prefix="/qa", tags=["QA Operations"])


config = get_config()
ollama_client = OllamaClient(
    base_url=config.ollama_url,
    model=config.ollama_model
)
embedder = Embedder()
qdrant_helper = QdrantHelper(
    url=config.qdrant_url,
    collection_name=config.qdrant_collection_name
)


@router.post("/save", response_model=SaneQAResponse)
async def save_qa_handler(body: SaveQABody, _: bool = Depends(get_current_user)) -> SaneQAResponse:
    try:
        dialog_text = "\n".join([msg.role + ": " + msg.content for msg in body.dialog])
        extracted_question, extracted_answer = ollama_client.extract_qa_pair(dialog_text)
        
        if not extracted_question or not extracted_answer:
            fallback_question = None
            fallback_answer = None
            
            for msg in body.dialog:
                if msg.role == RoleType.USER and "?" in msg.content and not fallback_question:
                    fallback_question = msg.content.strip()
                elif msg.role == RoleType.SUPPORT and fallback_question and not fallback_answer:
                    fallback_answer = msg.content.strip()
                    break
            
            if fallback_question and fallback_answer:
                extracted_question = fallback_question
                extracted_answer = fallback_answer
            else:
                return SaneQAResponse(
                    status="error", 
                    message="Failed to extract question or answer from dialog"
                )

        try:
            vector_question = embedder.encode(extracted_question)
        except Exception as e:
            raise EmbeddingException(f"Error creating embedding: {str(e)}")

        try:
            qdrant_helper.add_vector(vector_question, body.model_dump())
        except Exception as e:
            raise VectorStoreException(f"Error saving to vector store: {str(e)}")

        try:
            with get_db_context() as db:
                save_qa(
                    db=db,
                    ticket_id=body.ticket_id,
                    question=extracted_question,
                    answer=extracted_answer,
                    source=body.model_dump()
                )
        except Exception as e:
            raise DatabaseException(f"Error saving to database: {str(e)}")

        return SaneQAResponse(
            status="success",
            message="QA pair successfully saved",
            extracted_question=extracted_question,
            extracted_answer=extracted_answer
        )
        
    except (DatabaseException, LLMException, EmbeddingException, VectorStoreException):
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.post("/search", response_model=GetAnswerResponse)
async def get_answer_handler(body: GetAnswerBody, _: bool = Depends(get_current_user)) -> GetAnswerResponse:
    try:
        try:
            vector_question = embedder.encode(body.question)
        except Exception as e:
            raise EmbeddingException(f"Error creating embedding: {str(e)}")

        try:
            search_results = qdrant_helper.search_similar(vector_question, body.top_k)
        except Exception as e:
            raise VectorStoreException(f"Error searching in vector store: {str(e)}")

        try:
            with get_db_context() as db:
                ticket_ids = [result["ticket_id"] for result in search_results]
                qas = get_qa(db, ticket_ids)
                
                results = []
                for result in search_results:
                    qa = next((qa for qa in qas if qa.ticket_id == result["ticket_id"]), None)
                    if qa:
                        results.append(GetAnswerResultResponse(
                            question=qa.question,
                            answer=qa.answer,
                            similarity=result["score"],
                            ticket_id=int(qa.ticket_id)
                        ))
        except Exception as e:
            raise DatabaseException(f"Error getting data from database: {str(e)}")

        return GetAnswerResponse(
            query=body.question,
            results=results,
            total_found=len(results)
        )
        
    except (DatabaseException, EmbeddingException, VectorStoreException):
        raise

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
