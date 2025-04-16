from typing import List, Dict, Any
import weaviate
import logging
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv
from urllib.parse import urlparse

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

WEAVIATE_URL = os.getenv("WEAVIATE_URL")
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY")

if not WEAVIATE_URL:
    raise ValueError("WEAVIATE_URL environment variable is not set")

# Ensure URL has a scheme
if not urlparse(WEAVIATE_URL).scheme:
    WEAVIATE_URL = f"https://{WEAVIATE_URL}"


class RAGService:
    def __init__(self):
        logger.info(f"Connecting to Weaviate at {WEAVIATE_URL}")
        # Initialize Weaviate client with the correct v4 syntax
        auth_config = weaviate.auth.AuthApiKey(api_key=WEAVIATE_API_KEY)
        self.client = weaviate.connect_to_weaviate_cloud(
            cluster_url=WEAVIATE_URL,
            auth_credentials=auth_config,
            headers={"X-OpenAI-Api-Key": os.getenv("OPENAI_API_KEY")},
        )
        self.llm = ChatOpenAI(model="gpt-4-turbo-preview", temperature=0)

        self.qa_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are an expert consultant analyzing reports. 
            Use the provided context to answer the question.
            If you cannot answer from the context, say so.
            Include specific references to the source documents when possible.
            Be concise but informative.""",
                ),
                ("user", "Context:\n{context}\n\nQuestion: {question}"),
            ]
        )

    async def process_query(
        self, query: str, tenant_id: str, max_results: int = 3
    ) -> Dict[str, Any]:
        try:
            logger.info(f"Querying Weaviate for tenant {tenant_id}")

            # Query with tenant isolation
            response = (
                self.client.query.get("DocumentChunk")
                .with_tenant(tenant_id)
                .with_near_text({"concepts": [query]})
                .with_limit(max_results)
                .with_additional(["distance"])
                .do()
            )

            # Get objects from response
            objects = response.get("data", {}).get("Get", {}).get("DocumentChunk", [])
            logger.info(f"Retrieved {len(objects)} chunks")

            # Extract context from chunks
            context = [obj.get("content", "") for obj in objects]
            context_str = "\n".join(context)

            # Generate answer using LLM
            chain = self.qa_prompt | self.llm
            answer = await chain.ainvoke({"context": context_str, "question": query})

            return {
                "answer": answer.content,
                "sources": [
                    {
                        "document_id": obj.get("document_id", "Unknown"),
                        "metadata": obj.get("metadata", {}),
                        "distance": obj.get("_additional", {}).get("distance", 1.0),
                    }
                    for obj in objects
                ],
                "context": context,
            }

        except Exception as e:
            logger.error(f"Error in RAG service: {str(e)}", exc_info=True)
            raise
