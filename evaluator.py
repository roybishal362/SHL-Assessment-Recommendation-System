"""
SHL Assessment Recommendation System Evaluation Script (Improved)

This script evaluates the performance of the recommendation engine 
using standard information retrieval metrics including MAP@k and Recall@k.
"""

import os
import json
import numpy as np
import logging
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Tuple
from recommend_engine import SHLRecommendationEngine
import re

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RecommendationEvaluator:
    def __init__(self, test_data_path="data/test_data.json", 
                 recommendation_engine=None,
                 output_dir="evaluation_results"):
        """
        Initialize the evaluator with test data and recommendation engine.
        
        Args:
            test_data_path: Path to test data JSON file
            recommendation_engine: Instance of recommendation engine (if None, will create one)
            output_dir: Directory to save evaluation results
        """
        self.test_data_path = test_data_path
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Load test data
        self.test_data = self._load_test_data()
        
        # Initialize recommendation engine if not provided
        if recommendation_engine is None:
            logger.info("Initializing recommendation engine...")
            self.engine = SHLRecommendationEngine()
        else:
            self.engine = recommendation_engine
    
    def _normalize_assessment_name(self, name: str) -> str:
        """
        Normalize assessment name for better comparison.
        
        Args:
            name: Assessment name
            
        Returns:
            Normalized assessment name
        """
        # Remove "| SHL" suffix and trim
        name = re.sub(r'\|\s*SHL\s*$', '', name).strip()
        # Convert to lowercase and remove extra spaces
        name = re.sub(r'\s+', ' ', name.lower())
        return name
        
    def _load_test_data(self) -> Dict[str, Any]:
        """Load test data from JSON file or create default test set."""
        try:
            if hasattr(self, 'test_data_path') and self.test_data_path and os.path.exists(self.test_data_path):
                with open(self.test_data_path, 'r', encoding='utf-8') as f:
                    test_data = json.load(f)
                logger.info(f"Loaded test data from {self.test_data_path}")
                return test_data
            else:
                # Create a default test set based on assignment document
                logger.info("Test data file not found. Creating default test set...")
                default_test_data = self._create_default_test_data()
                
                # Save the default test set
                os.makedirs(os.path.dirname(self.test_data_path), exist_ok=True)
                with open(self.test_data_path, 'w', encoding='utf-8') as f:
                    json.dump(default_test_data, f, indent=4)
                
                logger.info(f"Created default test set with {len(default_test_data['queries'])} queries")
                return default_test_data
        except Exception as e:
            logger.error(f"Error loading test data: {e}")
            return {"queries": [], "ground_truth": {}}
            
    def _create_default_test_data(self) -> Dict[str, Any]:
        """Create default test data based on the assignment document."""
        # Test queries and ground truth from assignment document
        test_data = {
            "queries": [
                "I am hiring for Java developers who can also collaborate effectively with my business teams. Looking for an assessment(s) that can be completed in 40 minutes.",
                "I want to hire new graduates for a sales role in my company, the budget is for about an hour for each test. Give me some options",
                "I am looking for a COO for my company in China and I want to see if they are culturally a right fit for our company. Suggest me an assessment that they can complete in about an hour",
                "Content Writer required, expert in English and SEO.",
                "Find me 1 hour long assesment for the below job at SHL\nJob Description\nJoin a community that is shaping the future of work! SHL, People Science. People Answers.\nAre you a seasoned QA Engineer with a flair for innovation?...",
                "ICICI Bank Assistant Admin, Experience required 0-2 years, test should be 30-40 mins long"
            ],
            "ground_truth": {
                "I am hiring for Java developers who can also collaborate effectively with my business teams. Looking for an assessment(s) that can be completed in 40 minutes.": [
                    "Automata - Fix (New) | SHL",
                    "Core Java (Entry Level) (New) | SHL",
                    "Java 8 (New) | SHL",
                    "Core Java (Advanced Level) (New) | SHL",
                    "Agile Software Development | SHL",
                    "Technology Professional 8.0 Job Focused Assessment | SHL",
                    "Computer Science (New) | SHL"
                ],
                "I want to hire new graduates for a sales role in my company, the budget is for about an hour for each test. Give me some options": [
                    "Entry level Sales 7.1 (International) | SHL",
                    "Entry Level Sales Sift Out 7.1 | SHL",
                    "Entry Level Sales Solution | SHL",
                    "Sales Representative Solution | SHL",
                    "Sales Support Specialist Solution | SHL",
                    "Technical Sales Associate Solution | SHL",
                    "SVAR - Spoken English (Indian Accent) (New) | SHL",
                    "Sales & Service Phone Solution | SHL",
                    "Sales & Service Phone Simulation | SHL",
                    "English Comprehension (New) | SHL"
                ],
                "I am looking for a COO for my company in China and I want to see if they are culturally a right fit for our company. Suggest me an assessment that they can complete in about an hour": [
                    "Motivation Questionnaire MQM5 | SHL",
                    "Global Skills Assessment | SHL",
                    "Graduate 8.0 Job Focused Assessment | SHL"
                ],
                "Content Writer required, expert in English and SEO.": [
                    "Drupal (New) | SHL",
                    "Search Engine Optimization (New) | SHL",
                    "Administrative Professional - Short Form | SHL",
                    "Entry Level Sales Sift Out 7.1 | SHL",
                    "General Entry Level – Data Entry 7.0 Solution | SHL"
                ],
                "Find me 1 hour long assesment for the below job at SHL\nJob Description\nJoin a community that is shaping the future of work! SHL, People Science. People Answers.\nAre you a seasoned QA Engineer with a flair for innovation?...": [
                    "Automata Selenium | SHL",
                    "Automata - Fix (New) | SHL",
                    "Automata Front End | SHL",
                    "JavaScript (New) | SHL",
                    "HTML/CSS (New) | SHL",
                    "HTML5 (New) | SHL",
                    "CSS3 (New) | SHL",
                    "Selenium (New) | SHL",
                    "SQL Server (New) | SHL",
                    "Automata - SQL (New) | SHL",
                    "Manual Testing (New) | SHL"
                ],
                "ICICI Bank Assistant Admin, Experience required 0-2 years, test should be 30-40 mins long": [
                    "Administrative Professional - Short Form | SHL",
                    "Verify - Numerical Ability | SHL",
                    "Financial Professional - Short Form | SHL",
                    "Bank Administrative Assistant - Short Form | SHL",
                    "General Entry Level – Data Entry 7.0 Solution | SHL",
                    "Basic Computer Literacy (Windows 10) (New) | SHL"
                ]
            }
        }
        
        return test_data
        
    def _calculate_precision_at_k(self, recommended: List[str], relevant: List[str], k: int) -> float:
        """
        Calculate precision@k metric.
        
        Args:
            recommended: List of recommended assessment titles
            relevant: List of relevant assessment titles (ground truth)
            k: Number of recommendations to consider
            
        Returns:
            Precision@k value (float)
        """
        if not recommended or k <= 0:
            return 0.0
        
        # Normalize all titles for comparison
        norm_recommended = [self._normalize_assessment_name(title) for title in recommended[:k]]
        norm_relevant = [self._normalize_assessment_name(title) for title in relevant]
        
        # Count relevant items in top-k recommendations
        relevant_count = sum(1 for item in norm_recommended if item in norm_relevant)
        
        # Calculate precision@k
        return relevant_count / min(k, len(recommended))
    
    def _calculate_recall_at_k(self, recommended: List[str], relevant: List[str], k: int) -> float:
        """
        Calculate recall@k metric.
        
        Args:
            recommended: List of recommended assessment titles
            relevant: List of relevant assessment titles (ground truth)
            k: Number of recommendations to consider
            
        Returns:
            Recall@k value (float)
        """
        if not recommended or not relevant or k <= 0:
            return 0.0
        
        # Normalize all titles for comparison
        norm_recommended = [self._normalize_assessment_name(title) for title in recommended[:k]]
        norm_relevant = [self._normalize_assessment_name(title) for title in relevant]
        
        # Count relevant items in top-k recommendations
        relevant_count = sum(1 for item in norm_recommended if item in norm_relevant)
        
        # Calculate recall@k
        return relevant_count / len(relevant)
    
    def _calculate_ap_at_k(self, recommended: List[str], relevant: List[str], k: int) -> float:
        """
        Calculate average precision at k (AP@k).
        
        Args:
            recommended: List of recommended assessment titles
            relevant: List of relevant assessment titles (ground truth)
            k: Number of recommendations to consider
            
        Returns:
            AP@k value (float)
        """
        if not recommended or not relevant or k <= 0:
            return 0.0
        
        # Normalize all titles for comparison
        norm_recommended = [self._normalize_assessment_name(title) for title in recommended[:k]]
        norm_relevant = [self._normalize_assessment_name(title) for title in relevant]
        
        # Calculate average precision
        ap = 0.0
        relevant_count = 0
        
        for i, item in enumerate(norm_recommended, 1):
            # If current item is relevant
            if item in norm_relevant:
                relevant_count += 1
                # Add precision at current position
                precision_at_i = relevant_count / i
                ap += precision_at_i
        
        # Normalize by minimum of number of relevant items or k
        if relevant_count > 0:
            ap = ap / min(len(relevant), k)
        else:
            ap = 0.0
            
        return ap
    
    def evaluate(self, k_values: List[int] = [3, 5, 10], verbose: bool = True) -> Dict[str, Any]:
        """
        Evaluate the recommendation engine using multiple metrics at different k values.
        
        Args:
            k_values: List of k values to evaluate at
            verbose: Whether to print detailed results
            
        Returns:
            Dictionary of evaluation results
        """
        # Initialize results dictionary
        results = {
            "overall": {f"mean_precision@{k}": 0.0 for k in k_values},
            "per_query": {}
        }
        
        # Add other metrics to overall results
        for k in k_values:
            results["overall"][f"mean_recall@{k}"] = 0.0
            results["overall"][f"map@{k}"] = 0.0
        
        # Skip evaluation if no test data
        if not self.test_data or not self.test_data.get("queries") or not self.test_data.get("ground_truth"):
            logger.warning("No test data available for evaluation")
            return results
        
        # Initialize metric accumulators
        metrics_sum = {k: {"precision": 0.0, "recall": 0.0, "ap": 0.0} for k in k_values}
        
        # Track non-empty recommendations count
        valid_query_count = 0
        
        # Evaluate each query
        for query in self.test_data["queries"]:
            # Skip if no ground truth for this query
            if query not in self.test_data["ground_truth"]:
                logger.warning(f"No ground truth found for query: {query[:50]}...")
                continue
            
            # Get relevant assessments (ground truth)
            relevant_assessments = self.test_data["ground_truth"][query]
            
            # Get recommendations from engine
            max_k = max(k_values)
            recommendations = self.engine.recommend(query, top_k=max_k)
            
            # Skip if no recommendations
            if not recommendations:
                logger.warning(f"No recommendations returned for query: {query[:50]}...")
                continue
            
            # Extract recommendation titles - handle different possible formats
            recommended_titles = []
            for rec in recommendations:
                # Handle dictionary format
                if isinstance(rec, dict):
                    title = rec.get('title', '')
                    # If title doesn't end with "| SHL", add it
                    if "| SHL" not in title:
                        title = f"{title} | SHL"
                    recommended_titles.append(title)
                # Handle string format
                elif isinstance(rec, str):
                    title = rec
                    if "| SHL" not in title:
                        title = f"{title} | SHL"
                    recommended_titles.append(title)
            
            # Create query results
            query_result = {
                "query": query,
                "recommendations": recommendations,
                "recommended_titles": recommended_titles,
                "relevant_items": relevant_assessments,
                "metrics": {}
            }
            
            # Calculate metrics for each k value
            for k in k_values:
                # Calculate precision, recall, and AP at k
                precision_k = self._calculate_precision_at_k(recommended_titles, relevant_assessments, k)
                recall_k = self._calculate_recall_at_k(recommended_titles, relevant_assessments, k)
                ap_k = self._calculate_ap_at_k(recommended_titles, relevant_assessments, k)
                
                # Store metrics for this query
                query_result["metrics"][f"precision@{k}"] = precision_k
                query_result["metrics"][f"recall@{k}"] = recall_k
                query_result["metrics"][f"ap@{k}"] = ap_k
                
                # Accumulate metrics
                metrics_sum[k]["precision"] += precision_k
                metrics_sum[k]["recall"] += recall_k
                metrics_sum[k]["ap"] += ap_k
            
            # Add query result to per-query results
            results["per_query"][query] = query_result
            valid_query_count += 1
            
            # Print query results if verbose
            if verbose:
                print(f"\nQuery: {query[:100]}...")
                print(f"  Recommendations: {', '.join(recommended_titles[:3])}...")
                
                # Debug: Print title comparisons
                logger.debug("Ground truth titles (normalized):")
                for i, title in enumerate(relevant_assessments, 1):
                    norm_title = self._normalize_assessment_name(title)
                    logger.debug(f"  {i}. {title} -> {norm_title}")
                
                logger.debug("Recommended titles (normalized):")
                for i, title in enumerate(recommended_titles[:k_values[-1]], 1):
                    norm_title = self._normalize_assessment_name(title)
                    in_gt = self._normalize_assessment_name(title) in [self._normalize_assessment_name(t) for t in relevant_assessments]
                    logger.debug(f"  {i}. {title} -> {norm_title} (In ground truth: {in_gt})")
                
                for k in k_values:
                    print(f"  Precision@{k}: {query_result['metrics'][f'precision@{k}']:.4f}, "
                          f"Recall@{k}: {query_result['metrics'][f'recall@{k}']:.4f}, "
                          f"AP@{k}: {query_result['metrics'][f'ap@{k}']:.4f}")
        
        # Calculate mean metrics if we have valid queries
        if valid_query_count > 0:
            for k in k_values:
                results["overall"][f"mean_precision@{k}"] = metrics_sum[k]["precision"] / valid_query_count
                results["overall"][f"mean_recall@{k}"] = metrics_sum[k]["recall"] / valid_query_count
                results["overall"][f"map@{k}"] = metrics_sum[k]["ap"] / valid_query_count
        
        # Print overall results if verbose
        if verbose:
            print("\nOverall Evaluation Results:")
            for k in k_values:
                print(f"  Mean Precision@{k}: {results['overall'][f'mean_precision@{k}']:.4f}")
                print(f"  Mean Recall@{k}: {results['overall'][f'mean_recall@{k}']:.4f}")
                print(f"  MAP@{k}: {results['overall'][f'map@{k}']:.4f}")
        
        return results
    
    def generate_report(self, results: Dict[str, Any]) -> None:
        """
        Generate a detailed evaluation report and save to file.
        
        Args:
            results: Evaluation results dictionary
        """
        # Create report directory
        report_dir = os.path.join(self.output_dir, "report")
        os.makedirs(report_dir, exist_ok=True)
        
        # Generate summary report
        self._generate_summary_report(results, report_dir)
        
        # Generate detailed report
        self._generate_detailed_report(results, report_dir)
        
        # Generate visualizations
        self._generate_visualizations(results, report_dir)
        
        logger.info(f"Evaluation report generated in {report_dir}")
    
    def _generate_summary_report(self, results: Dict[str, Any], report_dir: str) -> None:
        """Generate summary evaluation report."""
        # Create summary report
        report_path = os.path.join(report_dir, "summary_report.md")
        
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("# SHL Assessment Recommendation System Evaluation\n\n")
            
            # Write overall metrics
            f.write("## Overall Metrics\n\n")
            f.write("| Metric | Value |\n")
            f.write("|--------|-------|\n")
            
            for metric, value in results["overall"].items():
                f.write(f"| {metric} | {value:.4f} |\n")
            
            # Write per-k metrics
            k_values = sorted([int(k.split("@")[1]) for k in results["overall"].keys() if k.startswith("map@")])
            
            f.write("\n## Metrics by k Value\n\n")
            f.write("| k | Precision | Recall | MAP |\n")
            f.write("|---|-----------|--------|-----|\n")
            
            for k in k_values:
                precision = results["overall"].get(f"mean_precision@{k}", 0.0)
                recall = results["overall"].get(f"mean_recall@{k}", 0.0)
                map_k = results["overall"].get(f"map@{k}", 0.0)
                
                f.write(f"| {k} | {precision:.4f} | {recall:.4f} | {map_k:.4f} |\n")
    
    def _generate_detailed_report(self, results: Dict[str, Any], report_dir: str) -> None:
        """Generate detailed per-query evaluation report."""
        # Create detailed report
        report_path = os.path.join(report_dir, "detailed_report.md")
        
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("# Detailed Query Evaluation\n\n")
            
            for query, query_result in results["per_query"].items():
                # Abbreviate long queries for readability
                query_short = query[:50] + "..." if len(query) > 50 else query
                f.write(f"## Query: {query_short}\n\n")
                
                # Write metrics
                f.write("### Metrics\n\n")
                f.write("| Metric | Value |\n")
                f.write("|--------|-------|\n")
                
                for metric, value in query_result["metrics"].items():
                    f.write(f"| {metric} | {value:.4f} |\n")
                
                # Write recommendations
                f.write("\n### Recommendations\n\n")
                for i, title in enumerate(query_result["recommended_titles"][:10], 1):
                    # Determine if recommendation is in ground truth
                    is_relevant = self._normalize_assessment_name(title) in [
                        self._normalize_assessment_name(gt) for gt in query_result["relevant_items"]
                    ]
                    relevance_mark = "✓" if is_relevant else "✗"
                    
                    # Format with URL if available in recommendations
                    if isinstance(query_result["recommendations"][i-1], dict) and "url" in query_result["recommendations"][i-1]:
                        url = query_result["recommendations"][i-1]["url"]
                        f.write(f"{i}. [{title}]({url}) {relevance_mark}\n")
                    else:
                        f.write(f"{i}. {title} {relevance_mark}\n")
                
                # Write relevant items
                f.write("\n### Relevant Items (Ground Truth)\n\n")
                for i, item in enumerate(query_result["relevant_items"], 1):
                    f.write(f"{i}. {item}\n")
                
                f.write("\n---\n\n")
    
    def _generate_visualizations(self, results: Dict[str, Any], report_dir: str) -> None:
        """Generate visualizations of evaluation results."""
        # Create visualizations directory
        viz_dir = os.path.join(report_dir, "visualizations")
        os.makedirs(viz_dir, exist_ok=True)
        
        # Extract k values
        k_values = sorted([int(k.split("@")[1]) for k in results["overall"].keys() if k.startswith("map@")])
        
        # Prepare data for plots
        metrics = {
            "Precision": [results["overall"].get(f"mean_precision@{k}", 0.0) for k in k_values],
            "Recall": [results["overall"].get(f"mean_recall@{k}", 0.0) for k in k_values],
            "MAP": [results["overall"].get(f"map@{k}", 0.0) for k in k_values]
        }
        
        # Create metrics by k plot
        plt.figure(figsize=(10, 6))
        
        for metric, values in metrics.items():
            plt.plot(k_values, values, marker='o', label=metric)
        
        plt.title("Evaluation Metrics by k Value")
        plt.xlabel("k (Number of Recommendations)")
        plt.ylabel("Metric Value")
        plt.xticks(k_values)
        plt.ylim(0, 1)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        plt.savefig(os.path.join(viz_dir, "metrics_by_k.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create per-query metrics plot
        if results["per_query"]:
            # Get per-query metrics for k=3
            queries = list(results["per_query"].keys())
            query_labels = [f"Q{i+1}" for i in range(len(queries))]
            
            precision_values = [results["per_query"][q]["metrics"].get("precision@3", 0.0) for q in queries]
            recall_values = [results["per_query"][q]["metrics"].get("recall@3", 0.0) for q in queries]
            ap_values = [results["per_query"][q]["metrics"].get("ap@3", 0.0) for q in queries]
            
            # Create bar chart
            plt.figure(figsize=(12, 6))
            width = 0.25
            x = np.arange(len(query_labels))
            
            plt.bar(x - width, precision_values, width, label='Precision@3')
            plt.bar(x, recall_values, width, label='Recall@3')
            plt.bar(x + width, ap_values, width, label='AP@3')
            
            plt.title("Per-Query Metrics (k=3)")
            plt.xlabel("Query")
            plt.ylabel("Metric Value")
            plt.xticks(x, query_labels)
            plt.ylim(0, 1)
            plt.grid(True, linestyle='--', alpha=0.7, axis='y')
            plt.legend()
            
            plt.savefig(os.path.join(viz_dir, "per_query_metrics.png"), dpi=300, bbox_inches='tight')
            plt.close()
            
            # Create mapping between query labels and actual queries
            query_mapping = {f"Q{i+1}": query for i, query in enumerate(queries)}
            
            # Save query mapping
            with open(os.path.join(viz_dir, "query_mapping.json"), "w", encoding="utf-8") as f:
                json.dump(query_mapping, f, indent=4)
            
    def run_optimization_experiments(self, experiment_configs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Run optimization experiments with different recommendation engine configurations.
        
        Args:
            experiment_configs: List of experiment configurations
                               (each is a dict of parameters for SHLRecommendationEngine)
                               
        Returns:
            Dictionary of experiment results
        """
        experiment_results = {}
        
        for i, config in enumerate(experiment_configs, 1):
            experiment_name = config.pop("name", f"Experiment_{i}")
            logger.info(f"Running experiment: {experiment_name}")
            
            # Initialize engine with configuration
            engine = SHLRecommendationEngine(**config)
            
            # Run evaluation
            results = self.evaluate(verbose=False)
            
            # Store results
            experiment_results[experiment_name] = {
                "config": config,
                "results": results["overall"]
            }
            
            # Print key metrics
            logger.info(f"  Results for {experiment_name}:")
            logger.info(f"    MAP@3: {results['overall']['map@3']:.4f}")
            logger.info(f"    Mean Recall@3: {results['overall']['mean_recall@3']:.4f}")
        
        return experiment_results
    
    def save_evaluation_results(self, results: Dict[str, Any]) -> None:
        """Save evaluation results to file."""
        results_path = os.path.join(self.output_dir, "evaluation_results.json")
        
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4)
            
        logger.info(f"Evaluation results saved to {results_path}")
        

if __name__ == "__main__":
    # Create evaluator
    evaluator = RecommendationEvaluator(
        test_data_path="data/test_data.json",
        output_dir="evaluation_results"
    )
    
    # Run evaluation
    results = evaluator.evaluate(k_values=[3, 5, 10])
    
    # Generate report
    evaluator.generate_report(results)
    
    # Save results
    evaluator.save_evaluation_results(results)
    
    # Example optimization experiments (uncomment to run)
    # experiment_configs = [
    #     {
    #         "name": "Default_Model",
    #         "model_name": "sentence-transformers/all-MiniLM-L6-v2"
    #     },
    #     {
    #         "name": "MPNet_Model",
    #         "model_name": "sentence-transformers/all-mpnet-base-v2"
    #     }
    # ]
    # optimization_results = evaluator.run_optimization_experiments(experiment_configs)