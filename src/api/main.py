from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import RAGService using absolute import from src directory
from api.services.rag import RAGService

app = FastAPI(title="Consulting Reports RAG API")
rag_service = RAGService()


class QueryRequest(BaseModel):
    query: str
    max_results: Optional[int] = 3


class QueryResponse(BaseModel):
    answer: str
    sources: List[dict]
    context: List[str]


@app.post("/query", response_model=QueryResponse)
async def query_documents(
    request: QueryRequest,
    tenant_id: str = Header(..., description="Tenant ID for multi-tenant access"),
):
    try:
        logger.info(f"Processing query for tenant {tenant_id}: {request.query}")
        result = await rag_service.process_query(
            query=request.query, tenant_id=tenant_id, max_results=request.max_results
        )
        logger.info("Query processed successfully")
        return result
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


if __name__ == "__main__":
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
