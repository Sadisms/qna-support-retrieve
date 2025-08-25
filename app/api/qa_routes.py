from fastapi import APIRouter, HTTPException, Depends

from app.core.database import get_db_context
from app.core.exceptions import DatabaseException, LLMException, EmbeddingException, VectorStoreException
from app.core.auth import get_current_workspace
from app.models.schemas import SaveQABody, SaneQAResponse, GetAnswerBody, GetAnswerResponse, GetAnswerResultResponse, RoleType
from app.services.qa_service import save_qa, get_qa, get_qa_by_ticket_id
from app.services.llm_client import OpenAIClient
from app.services.embeddings import Embedder
from app.services.qdrant import QdrantHelper
from app.core.config import get_config


router = APIRouter(prefix="/qa", tags=["QA Operations"])


config = get_config()


llm_client = OpenAIClient(
    api_key=config.openai_api_key,
    model=config.openai_model,
    proxy_url=config.openai_proxy_url,
)

embedder = Embedder()


def get_qdrant_helper(workspace_id: str) -> QdrantHelper:
    return QdrantHelper(
        url=config.qdrant_url,
        collection_name=config.qdrant_collection_name,
        workspace_id=workspace_id
    )


@router.post("/save", response_model=SaneQAResponse)
async def save_qa_handler(body: SaveQABody, workspace_id: str = Depends(get_current_workspace)) -> SaneQAResponse:
    try:
        try:
            with get_db_context() as db:
                existing = get_qa_by_ticket_id(db, workspace_id, body.ticket_id)
                if existing:
                    return SaneQAResponse(
                        status="success",
                        message="Ticket already saved",
                        extracted_question=existing.question,
                        extracted_answer=existing.answer,
                        ticket_id=int(existing.ticket_id),
                        already_saved=True
                    )
        except Exception as e:
            raise DatabaseException(f"Error checking existing ticket: {str(e)}")

        full_dialog_text = "\n".join([
            (("USER" if msg.role == RoleType.USER else "SUPPORT") + ": " + msg.content)
            for msg in body.dialog
        ])

        qa_result = llm_client.extract_qa_pair_with_validation(full_dialog_text)
            
        if not qa_result:
            return SaneQAResponse(
                status="error",
                message="No high-quality Q&A pair found in dialog. Dialog may not contain business-relevant questions or clear answers."
            )
        
        extracted_question = qa_result["question"]
        extracted_answer = qa_result["answer"]

        try:
            vector_question = embedder.encode(extracted_question)
        except Exception as e:
            raise EmbeddingException(f"Error creating embedding: {str(e)}")

        try:
            qdrant_helper = get_qdrant_helper(workspace_id)
            qdrant_helper.add_vector(vector_question, body.model_dump())
        except Exception as e:
            raise VectorStoreException(f"Error saving to vector store: {str(e)}")

        try:
            with get_db_context() as db:
                save_qa(
                    db=db,
                    workspace_id=workspace_id,
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
            extracted_answer=extracted_answer,
            ticket_id=int(body.ticket_id),
            already_saved=False
        )
        
    except (DatabaseException, LLMException, EmbeddingException, VectorStoreException):
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.post("/search", response_model=GetAnswerResponse)
async def get_answer_handler(body: GetAnswerBody, workspace_id: str = Depends(get_current_workspace)) -> GetAnswerResponse:
    try:
        try:
            vector_question = embedder.encode(body.question)
        except Exception as e:
            raise EmbeddingException(f"Error creating embedding: {str(e)}")

        try:
            qdrant_helper = get_qdrant_helper(workspace_id)
            search_results = qdrant_helper.search_similar(vector_question, body.top_k)
        except Exception as e:
            raise VectorStoreException(f"Error searching in vector store: {str(e)}")

        try:
            with get_db_context() as db:
                ticket_ids = [result["ticket_id"] for result in search_results]
                qas = get_qa(db, workspace_id, ticket_ids)
                
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


@router.get("/metrics")
async def get_performance_metrics(_: str = Depends(get_current_workspace)):
    """Get performance metrics for Q&A extraction (OpenAI only)"""
    
    stats = llm_client.get_performance_stats()
    
    if not stats:
        return {
            "message": "No performance data available yet",
            "total_extractions": 0
        }
    
    # Add cost comparison and recommendations
    response = {
        **stats,
        "cost_analysis": {
            "model_used": config.openai_model,
            "cost_per_extraction": stats.get('avg_cost_per_extraction', 0),
            "estimated_monthly_cost_1k_extractions": stats.get('avg_cost_per_extraction', 0) * 1000,
        },
        "performance_status": {
            "processing_time": "✅ Good" if stats.get('avg_processing_time', 0) < 2.0 else "⚠️ Slow",
            "success_rate": "✅ Excellent" if stats.get('success_rate', 0) > 0.95 else "⚠️ Needs attention",
            "cost_efficiency": "✅ Optimal" if stats.get('avg_cost_per_extraction', 0) < 0.005 else "⚠️ High cost"
        },
        "recommendations": []
    }
    
    # Add recommendations based on metrics
    if stats.get('avg_processing_time', 0) > 2.0:
        response["recommendations"].append("Consider prompt optimization to reduce processing time")
    
    if stats.get('success_rate', 1) < 0.95:
        response["recommendations"].append("Review error patterns to improve API success rate")
    
    if stats.get('avg_cost_per_extraction', 0) > 0.005:
        response["recommendations"].append("Consider token optimization to reduce costs")
    
    if not response["recommendations"]:
        response["recommendations"] = ["All metrics within optimal ranges"]
    
    return response
