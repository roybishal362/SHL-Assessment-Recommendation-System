from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, List, Optional, Any
import uvicorn
import logging
from recommend_engine import SHLRecommendationEngine
from pydantic import BaseModel
import requests
from bs4 import BeautifulSoup

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize recommendation engine
recommendation_engine = SHLRecommendationEngine()

# Create FastAPI app
app = FastAPI(
    title="SHL Assessment Recommendation API",
    description="API for recommending SHL assessments based on job descriptions or natural language queries",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

class QueryRequest(BaseModel):
    query: str
    max_results: Optional[int] = 10
    url: Optional[str] = None

class AssessmentResponse(BaseModel):
    title: str
    url: str
    remote_testing_support: str
    adaptive_irt_support: str
    duration: str
    test_type: str

class RecommendationResponse(BaseModel):
    recommendations: List[AssessmentResponse]
    query: str
    source: str  # 'text' or 'url'

@app.get("/health")
async def health_check():
    """
    Health check endpoint to verify the API is running.
    """
    return {"status": "ok", "message": "SHL Assessment Recommendation API is running"}

@app.post("/recommend", response_model=RecommendationResponse)
async def recommend_assessments(request: QueryRequest):
    """
    Recommend SHL assessments based on a job description or natural language query.
    """
    try:
        query = request.query
        source = "text"
        
        # If URL is provided, fetch the content
        if request.url:
            try:
                logger.info(f"Fetching content from URL: {request.url}")
                response = requests.get(request.url, timeout=10)
                response.raise_for_status()
                
                # Parse HTML content
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Extract text content
                text_content = soup.get_text(separator=" ", strip=True)
                
                # Use the extracted text as query
                query = text_content
                source = "url"
                
                logger.info(f"Successfully extracted content from URL: {request.url}")
            except Exception as e:
                logger.error(f"Error fetching URL content: {e}")
                raise HTTPException(status_code=400, detail=f"Failed to fetch content from URL: {str(e)}")
        
        # Get recommendations
        max_results = min(request.max_results, 10)  # Limit to 10 maximum
        recommendations = recommendation_engine.recommend_with_auto_filter(query, top_k=max_results)
        
        # Format response
        formatted_recommendations = []
        for rec in recommendations:
            formatted_recommendations.append(
                AssessmentResponse(
                    title=rec["title"],
                    url=rec["url"],
                    remote_testing_support=rec["remote_testing_support"],
                    adaptive_irt_support=rec["adaptive_irt_support"],
                    duration=rec["duration"],
                    test_type=rec["test_type"]
                )
            )
        
        return RecommendationResponse(
            recommendations=formatted_recommendations,
            query=request.query if source == "text" else f"Content from {request.url}",
            source=source
        )
        
    except Exception as e:
        logger.error(f"Error processing recommendation request: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
