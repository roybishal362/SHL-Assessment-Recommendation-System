import os
import json
import logging
import numpy as np
from typing import List, Dict, Any, Tuple
import pickle
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import time
import torch
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
import joblib
import regex as re

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SHLRecommendationEngine:
    def __init__(self, data_path="data/shl_assessments.json", 
                 embeddings_path="data/embeddings.pkl",
                 faiss_index_path="data/faiss_index",
                 model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.data_path = data_path
        self.embeddings_path = embeddings_path
        self.faiss_index_path = faiss_index_path
        self.model_name = model_name
        self.assessments = []
        self.vectorstore = None
        
        # Ensure directories exist
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        
        # Initialize embedding model
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Load assessments
        self._load_assessments()
        
        # Initialize vector store
        self._initialize_vector_store()
        
    def _load_assessments(self):
        """Load assessment data from JSON file."""
        try:
            if os.path.exists(self.data_path):
                with open(self.data_path, 'r', encoding='utf-8') as f:
                    self.assessments = json.load(f)
                logger.info(f"Loaded {len(self.assessments)} assessments from {self.data_path}")
            else:
                logger.warning(f"Assessment data file {self.data_path} not found")
        except Exception as e:
            logger.error(f"Error loading assessments: {e}")
    
    def _initialize_vector_store(self):
        """Initialize the vector store with assessment data."""
        try:
            # Check if FAISS index already exists
            if os.path.exists(f"{self.faiss_index_path}.faiss"):
                logger.info("Loading existing FAISS index...")
                self.vectorstore = FAISS.load_local(
                    self.faiss_index_path,
                    self.embedding_model
                )
                logger.info("FAISS index loaded successfully")
            else:
                logger.info("Creating new FAISS index...")
                self._create_vector_store()
        except Exception as e:
            logger.error(f"Error initializing vector store: {e}")
            # Try to create a new one if loading fails
            self._create_vector_store()
    
    def _create_vector_store(self):
        """Create a new vector store from assessment data."""
        try:
            # Prepare documents for indexing
            documents = []
            
            for assessment in self.assessments:
                # Create a rich text representation for indexing
                content = f"""
                Title: {assessment['title']}
                Description: {assessment.get('description', '')}
                Type: {assessment.get('test_type', 'N/A')}
                Duration: {assessment.get('duration', 'N/A')}
                Remote Testing: {assessment.get('remote_testing_support', 'No')}
                Adaptive Testing: {assessment.get('adaptive_irt_support', 'No')}
                Features: {', '.join(assessment.get('features', []))}
                """
                
                # Create a Document object
                doc = Document(
                    page_content=content,
                    metadata={
                        "title": assessment['title'],
                        "url": assessment['url'],
                        "remote_testing_support": assessment.get('remote_testing_support', 'No'),
                        "adaptive_irt_support": assessment.get('adaptive_irt_support', 'No'),
                        "duration": assessment.get('duration', 'N/A'),
                        "test_type": assessment.get('test_type', 'N/A')
                    }
                )
                
                documents.append(doc)
            
            # Create FAISS index
            self.vectorstore = FAISS.from_documents(
                documents,
                self.embedding_model
            )
            
            # Save the index
            self.vectorstore.save_local(self.faiss_index_path)
            logger.info(f"Created and saved FAISS index with {len(documents)} documents")
            
        except Exception as e:
            logger.error(f"Error creating vector store: {e}")
    
    def recommend(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Get assessment recommendations based on a query.
        
        Args:
            query: The query text (job description or natural language query)
            top_k: Maximum number of recommendations to return
            
        Returns:
            List of recommended assessments
        """
        try:
            # Ensure we have a valid vector store
            if not self.vectorstore:
                logger.warning("Vector store not initialized. Attempting to initialize...")
                self._initialize_vector_store()
                
                if not self.vectorstore:
                    logger.error("Failed to initialize vector store")
                    return []
            
            # Get relevant documents
            relevant_docs = self.vectorstore.similarity_search_with_score(query, k=top_k)
            
            # Convert to recommendations
            recommendations = []
            for doc, score in relevant_docs:
                recommendations.append({
                    "title": doc.metadata["title"],
                    "url": doc.metadata["url"],
                    "remote_testing_support": doc.metadata["remote_testing_support"],
                    "adaptive_irt_support": doc.metadata["adaptive_irt_support"],
                    "duration": doc.metadata["duration"],
                    "test_type": doc.metadata["test_type"],
                    "similarity_score": float(score)
                })
                
            return recommendations
            
        except Exception as e:
            logger.error(f"Error during recommendation: {e}")
            return []

    def filter_recommendations(self, recommendations: List[Dict], 
                              duration_limit: int = None,
                              remote_testing: bool = None,
                              adaptive_testing: bool = None,
                              test_type: str = None) -> List[Dict]:
        """
        Filter recommendations based on specified criteria.
        
        Args:
            recommendations: List of recommendation dictionaries
            duration_limit: Maximum duration in minutes
            remote_testing: Whether remote testing is required
            adaptive_testing: Whether adaptive testing is required
            test_type: Specific test type
            
        Returns:
            Filtered list of recommendations
        """
        filtered = recommendations.copy()
        
        if duration_limit:
            filtered = [
                rec for rec in filtered 
                if self._extract_duration_minutes(rec.get("duration", "N/A")) <= duration_limit
            ]
            
        if remote_testing is not None:
            required_value = "Yes" if remote_testing else "No"
            filtered = [
                rec for rec in filtered 
                if rec.get("remote_testing_support", "No") == required_value
            ]
            
        if adaptive_testing is not None:
            required_value = "Yes" if adaptive_testing else "No"
            filtered = [
                rec for rec in filtered 
                if rec.get("adaptive_irt_support", "No") == required_value
            ]
            
        if test_type:
            filtered = [
                rec for rec in filtered 
                if test_type.lower() in rec.get("test_type", "").lower()
            ]
            
        return filtered
    
    def _extract_duration_minutes(self, duration_str: str) -> int:
        """Extract minutes from duration string."""
        try:
            # Try to find numbers in the duration string
            import re
            numbers = re.findall(r'\d+', duration_str)
            if numbers:
                return int(numbers[0])
            else:
                return float('inf')  # Return infinity if no number found
        except:
            return float('inf')

    def extract_filters_from_query(self, query: str) -> Dict[str, Any]:
        """
        Extract filter criteria from a natural language query.
        
        Args:
            query: The natural language query
            
        Returns:
            Dictionary of filter criteria
        """
        filters = {
            "duration_limit": None,
            "remote_testing": None,
            "adaptive_testing": None,
            "test_type": None
        }
        
        # Extract duration limit
        duration_patterns = [
            r'(\d+)\s*min',
            r'(\d+)\s*minute',
            r'under\s*(\d+)',
            r'less than\s*(\d+)',
            r'within\s*(\d+)',
            r'max.*?(\d+)',
            r'maximum.*?(\d+)'
        ]
        
        for pattern in duration_patterns:
            match = re.search(pattern, query.lower())
            if match:
                filters["duration_limit"] = int(match.group(1))
                break
                
        # Check for remote testing requirement
        if re.search(r'remote|online|virtual', query.lower()):
            filters["remote_testing"] = True
            
        # Check for adaptive testing requirement
        if re.search(r'adaptive|irt|item response', query.lower()):
            filters["adaptive_testing"] = True
            
        # Extract test types
        test_types = ["cognitive", "personality", "behavioral", "situational", 
                     "technical", "aptitude", "skills", "java", "python", "sql", 
                     "sales", "leadership", "management", "english", "verbal", 
                     "numerical", "reasoning"]
                     
        for test_type in test_types:
            if test_type in query.lower():
                filters["test_type"] = test_type
                break
                
        return filters

    def recommend_with_auto_filter(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Get filtered recommendations based on query and automatically extracted filters.
        
        Args:
            query: The query text
            top_k: Maximum number of recommendations
            
        Returns:
            List of filtered recommendations
        """
        # Get initial recommendations
        recommendations = self.recommend(query, top_k=min(top_k * 2, 30))  # Get more than needed for filtering
        
        # Extract filters from query
        filters = self.extract_filters_from_query(query)
        
        # Apply filters
        filtered_recommendations = self.filter_recommendations(
            recommendations,
            duration_limit=filters["duration_limit"],
            remote_testing=filters["remote_testing"],
            adaptive_testing=filters["adaptive_testing"],
            test_type=filters["test_type"]
        )
        
        # If filtering resulted in too few results, return original recommendations
        if len(filtered_recommendations) < min(top_k, 1):
            return recommendations[:top_k]
        
        return filtered_recommendations[:top_k]


if __name__ == "__main__":
    # Test the recommendation engine
    engine = SHLRecommendationEngine()
    
    test_query = "Java developers who can collaborate with business teams"
    recommendations = engine.recommend_with_auto_filter(test_query, top_k=5)
    
    print(f"Query: {test_query}")
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec['title']} - {rec['url']}")
        print(f"   Remote Testing: {rec['remote_testing_support']}, Adaptive: {rec['adaptive_irt_support']}")
        print(f"   Duration: {rec['duration']}, Type: {rec['test_type']}")
        print()
