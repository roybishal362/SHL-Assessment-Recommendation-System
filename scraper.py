import requests
from bs4 import BeautifulSoup
import json
import time
import os
import re
import logging
import random
from urllib.parse import urlparse, parse_qs

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SHLCatalogScraper:
    def __init__(self, catalog_url="https://www.shl.com/solutions/products/product-catalog/", 
                 output_file="data/shl_assessments.json"):
        self.catalog_url = catalog_url
        self.output_file = output_file
        self.assessments = []
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Cache-Control': 'max-age=0',
        }
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
    def scrape_catalog_pages(self, max_pages=50):
        """Scrape all pages in the SHL product catalog."""
        logger.info(f"Starting scraping of SHL product catalog from {self.catalog_url}")
        
        try:
            page = 1
            while page <= max_pages:
                logger.info(f"Scraping page {page}")
                
                # Get the current page
                if page == 1:
                    url = self.catalog_url
                else:
                    url = f"{self.catalog_url}page/{page}/"
                
                response = requests.get(url, headers=self.headers)
                
                if response.status_code != 200:
                    logger.error(f"Failed to fetch page {page}: Status code {response.status_code}")
                    break
                
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Extract products from current page
                products = soup.select('.product-list .product-card')
                
                if not products:
                    logger.info(f"No products found on page {page}, stopping pagination")
                    break
                
                for product in products:
                    try:
                        # Extract basic info
                        title_element = product.select_one('h3')
                        if title_element:
                            title = title_element.text.strip()
                        else:
                            continue
                        
                        # Get URL
                        link_element = product.select_one('a')
                        if link_element and 'href' in link_element.attrs:
                            url = link_element['href']
                        else:
                            continue
                        
                        # Access the product detail page to get more info
                        assessment_details = self._get_assessment_details(url)
                        
                        # Create assessment object
                        assessment = {
                            "title": title,
                            "url": url,
                            **assessment_details
                        }
                        
                        # Add to the collection
                        self.assessments.append(assessment)
                        logger.info(f"Added assessment: {title}")
                        
                    except Exception as e:
                        logger.error(f"Error extracting product info: {e}")
                
                # Check if there's a next page by looking for pagination
                next_page = soup.select_one('a.page-numbers.next')
                if not next_page:
                    logger.info("No more pages found")
                    break
                
                page += 1
                
                # Sleep to avoid overwhelming the server
                time.sleep(random.uniform(1, 3))
            
            logger.info(f"Finished scraping. Collected {len(self.assessments)} assessments.")
            self._save_to_json()
            
        except Exception as e:
            logger.error(f"Error during scraping: {e}")
    
    def _get_assessment_details(self, url):
        """Get detailed information about an assessment by visiting its page."""
        details = {
            "remote_testing_support": "No",
            "adaptive_irt_support": "No",
            "duration": "N/A",
            "test_type": "N/A",
            "description": "",
            "features": []
        }
        
        try:
            # Add a small delay to avoid rate limiting
            time.sleep(random.uniform(0.5, 2))
            
            response = requests.get(url, headers=self.headers)
            
            if response.status_code != 200:
                logger.error(f"Failed to fetch details for {url}: Status code {response.status_code}")
                return details
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract description
            description_element = soup.select_one('.product-description, .description')
            if description_element:
                details["description"] = description_element.text.strip()
            
            # Extract features and specifications
            feature_elements = soup.select('.product-features li, .features li')
            details["features"] = [feature.text.strip() for feature in feature_elements]
                
            # Look for specifications
            spec_elements = soup.select('.product-specs tr, .specifications tr, .details tr')
            
            for row in spec_elements:
                cells = row.select('td')
                if len(cells) >= 2:
                    key = cells[0].text.strip().lower()
                    value = cells[1].text.strip()
                    
                    if "remote" in key or "online" in key:
                        details["remote_testing_support"] = "Yes" if "yes" in value.lower() else "No"
                    elif "adaptive" in key or "irt" in key:
                        details["adaptive_irt_support"] = "Yes" if "yes" in value.lower() else "No"
                    elif "duration" in key or "time" in key:
                        details["duration"] = value
                    elif "type" in key:
                        details["test_type"] = value
            
            # Extract from page text
            page_text = soup.get_text().lower()
            
            # Check for remote testing keywords
            if "remote testing" in page_text or "online assessment" in page_text or "remote proctoring" in page_text:
                details["remote_testing_support"] = "Yes"
                
            # Check for adaptive testing keywords
            if "adaptive testing" in page_text or " irt " in page_text or "item response theory" in page_text:
                details["adaptive_irt_support"] = "Yes"
                
            # Try to find duration
            if details["duration"] == "N/A":
                duration_patterns = [
                    r'(\d+)\s*(?:minute|min)',
                    r'duration[:\s]*(\d+)',
                    r'takes\s*(?:about)?\s*(\d+)'
                ]
                
                for pattern in duration_patterns:
                    match = re.search(pattern, page_text)
                    if match:
                        details["duration"] = f"{match.group(1)} minutes"
                        break
            
            # Try to find test type
            if details["test_type"] == "N/A":
                test_types = ["cognitive", "personality", "behavioral", "situational", "technical", "aptitude", "skills"]
                for test_type in test_types:
                    if test_type in page_text:
                        details["test_type"] = test_type.capitalize()
                        break
            
        except Exception as e:
            logger.error(f"Error fetching assessment details for {url}: {e}")
                
        return details
    
    def _save_to_json(self):
        """Save the collected assessments to a JSON file."""
        try:
            with open(self.output_file, 'w', encoding='utf-8') as f:
                json.dump(self.assessments, f, indent=4)
            logger.info(f"Saved {len(self.assessments)} assessments to {self.output_file}")
        except Exception as e:
            logger.error(f"Error saving to JSON: {e}")
    
    def load_from_json(self):
        """Load assessments from the JSON file."""
        try:
            if os.path.exists(self.output_file):
                with open(self.output_file, 'r', encoding='utf-8') as f:
                    self.assessments = json.load(f)
                logger.info(f"Loaded {len(self.assessments)} assessments from {self.output_file}")
                return True
            else:
                logger.warning(f"JSON file {self.output_file} not found")
                return False
        except Exception as e:
            logger.error(f"Error loading from JSON: {e}")
            return False

# Mock data generator to use in case scraping fails
def generate_mock_data(output_file="data/shl_assessments.json"):
    logger.info("Generating mock assessment data")
    assessments = [
        {
            "title": "Automata - Fix (New)",
            "url": "https://www.shl.com/solutions/products/product-catalog/view/automata-fix-new/",
            "remote_testing_support": "Yes",
            "adaptive_irt_support": "No",
            "duration": "35 minutes",
            "test_type": "Technical",
            "description": "Advanced code debugging assessment for experienced developers."
        },
        {
            "title": "Core Java (Entry Level) (New)",
            "url": "https://www.shl.com/solutions/products/product-catalog/view/core-java-entry-level-new/",
            "remote_testing_support": "Yes",
            "adaptive_irt_support": "No",
            "duration": "40 minutes",
            "test_type": "Technical",
            "description": "Assessment to test entry-level Java programming skills."
        },
        {
            "title": "Java 8 (New)",
            "url": "https://www.shl.com/solutions/products/product-catalog/view/java-8-new/",
            "remote_testing_support": "Yes",
            "adaptive_irt_support": "No",
            "duration": "45 minutes",
            "test_type": "Technical",
            "description": "Assessment for Java 8 programming language skills."
        },
        {
            "title": "Entry level Sales 7.1 (International)",
            "url": "https://www.shl.com/solutions/products/product-catalog/view/entry-level-sales-7-1/",
            "remote_testing_support": "Yes",
            "adaptive_irt_support": "No",
            "duration": "50 minutes",
            "test_type": "Behavioral",
            "description": "Assessment for entry-level sales positions."
        },
        {
            "title": "Motivation Questionnaire MQM5",
            "url": "https://www.shl.com/solutions/products/product-catalog/view/motivation-questionnaire-mqm5/",
            "remote_testing_support": "Yes",
            "adaptive_irt_support": "No",
            "duration": "25 minutes",
            "test_type": "Personality",
            "description": "Assesses what motivates and drives an individual at work."
        },
        {
            "title": "Search Engine Optimization (New)",
            "url": "https://www.shl.com/solutions/products/product-catalog/view/search-engine-optimization-new/",
            "remote_testing_support": "Yes",
            "adaptive_irt_support": "No",
            "duration": "30 minutes",
            "test_type": "Technical",
            "description": "Assessment for SEO knowledge and best practices."
        },
        {
            "title": "Automata Selenium",
            "url": "https://www.shl.com/solutions/products/product-catalog/view/automata-selenium/",
            "remote_testing_support": "Yes",
            "adaptive_irt_support": "No",
            "duration": "60 minutes",
            "test_type": "Technical",
            "description": "Assessment for Selenium automation testing skills."
        },
        {
            "title": "Administrative Professional - Short Form",
            "url": "https://www.shl.com/solutions/products/product-catalog/view/administrative-professional-short-form/",
            "remote_testing_support": "Yes",
            "adaptive_irt_support": "No",
            "duration": "30 minutes",
            "test_type": "Skills",
            "description": "Assessment for administrative professional skills."
        },
        {
            "title": "Verify - Verbal Ability - Next Generation",
            "url": "https://www.shl.com/solutions/products/product-catalog/view/verify-verbal-ability-next-generation/",
            "remote_testing_support": "Yes",
            "adaptive_irt_support": "Yes",
            "duration": "20 minutes",
            "test_type": "Cognitive",
            "description": "Measures verbal reasoning skills."
        },
        {
            "title": "Occupational Personality Questionnaire OPQ32r",
            "url": "https://www.shl.com/solutions/products/product-catalog/view/occupational-personality-questionnaire-opq32r/",
            "remote_testing_support": "Yes", 
            "adaptive_irt_support": "No",
            "duration": "40 minutes",
            "test_type": "Personality",
            "description": "Comprehensive personality assessment for workplace behaviors."
        },
        {
            "title": "Core Java (Advanced Level) (New)",
            "url": "https://www.shl.com/solutions/products/product-catalog/view/core-java-advanced-level-new/",
            "remote_testing_support": "Yes",
            "adaptive_irt_support": "No",
            "duration": "50 minutes",
            "test_type": "Technical",
            "description": "Advanced Java programming skills assessment."
        },
        {
            "title": "Agile Software Development",
            "url": "https://www.shl.com/solutions/products/product-catalog/view/agile-software-development/",
            "remote_testing_support": "Yes",
            "adaptive_irt_support": "No",
            "duration": "30 minutes",
            "test_type": "Technical",
            "description": "Assessment for agile software development methodologies."
        },
        {
            "title": "Technology Professional 8.0 Job Focused Assessment",
            "url": "https://www.shl.com/solutions/products/product-catalog/view/technology-professional-8-0-job-focused-assessment/",
            "remote_testing_support": "Yes",
            "adaptive_irt_support": "Yes",
            "duration": "45 minutes",
            "test_type": "Technical",
            "description": "Comprehensive assessment for technology professionals."
        },
        {
            "title": "Computer Science (New)",
            "url": "https://www.shl.com/solutions/products/product-catalog/view/computer-science-new/",
            "remote_testing_support": "Yes",
            "adaptive_irt_support": "No",
            "duration": "40 minutes",
            "test_type": "Technical",
            "description": "Assessment for computer science fundamentals."
        },
        {
            "title": "Entry Level Sales Sift Out 7.1",
            "url": "https://www.shl.com/solutions/products/product-catalog/view/entry-level-sales-sift-out-7-1/",
            "remote_testing_support": "Yes",
            "adaptive_irt_support": "No",
            "duration": "30 minutes",
            "test_type": "Behavioral",
            "description": "Screening assessment for entry-level sales positions."
        },
        {
            "title": "Entry Level Sales Solution",
            "url": "https://www.shl.com/solutions/products/product-catalog/view/entry-level-sales-solution/",
            "remote_testing_support": "Yes",
            "adaptive_irt_support": "No",
            "duration": "55 minutes",
            "test_type": "Behavioral",
            "description": "Comprehensive solution for entry-level sales candidates."
        },
        {
            "title": "Sales Representative Solution",
            "url": "https://www.shl.com/solutions/products/product-catalog/view/sales-representative-solution/",
            "remote_testing_support": "Yes",
            "adaptive_irt_support": "No",
            "duration": "60 minutes",
            "test_type": "Behavioral",
            "description": "Comprehensive solution for sales representative candidates."
        },
        {
            "title": "Sales Support Specialist Solution",
            "url": "https://www.shl.com/solutions/products/product-catalog/view/sales-support-specialist-solution/",
            "remote_testing_support": "Yes",
            "adaptive_irt_support": "No",
            "duration": "45 minutes",
            "test_type": "Behavioral",
            "description": "Assessment for sales support specialist positions."
        },
        {
            "title": "Technical Sales Associate Solution",
            "url": "https://www.shl.com/solutions/products/product-catalog/view/technical-sales-associate-solution/",
            "remote_testing_support": "Yes",
            "adaptive_irt_support": "No",
            "duration": "60 minutes",
            "test_type": "Technical",
            "description": "Assessment for technical sales positions."
        },
        {
            "title": "Global Skills Assessment",
            "url": "https://www.shl.com/solutions/products/product-catalog/view/global-skills-assessment/",
            "remote_testing_support": "Yes",
            "adaptive_irt_support": "Yes",
            "duration": "60 minutes",
            "test_type": "Cognitive",
            "description": "Assessment for global skills and cultural fit."
        },
        {
            "title": "Graduate 8.0 Job Focused Assessment",
            "url": "https://www.shl.com/solutions/products/product-catalog/view/graduate-8-0-job-focused-assessment-4228/",
            "remote_testing_support": "Yes",
            "adaptive_irt_support": "No",
            "duration": "45 minutes",
            "test_type": "Cognitive",
            "description": "Comprehensive assessment for graduate positions."
        },
        {
            "title": "Drupal (New)",
            "url": "https://www.shl.com/solutions/products/product-catalog/view/drupal-new/",
            "remote_testing_support": "Yes",
            "adaptive_irt_support": "No",
            "duration": "35 minutes",
            "test_type": "Technical",
            "description": "Assessment for Drupal CMS skills."
        },
        {
            "title": "General Entry Level – Data Entry 7.0 Solution",
            "url": "https://www.shl.com/solutions/products/product-catalog/view/general-entry-level-data-entry-7-0-solution/",
            "remote_testing_support": "Yes",
            "adaptive_irt_support": "No",
            "duration": "30 minutes",
            "test_type": "Skills",
            "description": "Assessment for data entry skills."
        },
        {
            "title": "Basic Computer Literacy (Windows 10) (New)",
            "url": "https://www.shl.com/solutions/products/product-catalog/view/basic-computer-literacy-windows-10-new/",
            "remote_testing_support": "Yes",
            "adaptive_irt_support": "No",
            "duration": "25 minutes",
            "test_type": "Skills",
            "description": "Assessment for basic computer literacy skills."
        },
        {
            "title": "Automata Front End",
            "url": "https://www.shl.com/solutions/products/product-catalog/view/automata-front-end/",
            "remote_testing_support": "Yes",
            "adaptive_irt_support": "No",
            "duration": "60 minutes",
            "test_type": "Technical",
            "description": "Assessment for front-end development skills."
        },
        {
            "title": "JavaScript (New)",
            "url": "https://www.shl.com/solutions/products/product-catalog/view/javascript-new/",
            "remote_testing_support": "Yes",
            "adaptive_irt_support": "No",
            "duration": "40 minutes",
            "test_type": "Technical",
            "description": "Assessment for JavaScript programming skills."
        },
        {
            "title": "HTML/CSS (New)",
            "url": "https://www.shl.com/solutions/products/product-catalog/view/htmlcss-new/",
            "remote_testing_support": "Yes",
            "adaptive_irt_support": "No",
            "duration": "35 minutes",
            "test_type": "Technical",
            "description": "Assessment for HTML and CSS skills."
        },
        {
            "title": "HTML5 (New)",
            "url": "https://www.shl.com/solutions/products/product-catalog/view/html5-new/",
            "remote_testing_support": "Yes",
            "adaptive_irt_support": "No",
            "duration": "30 minutes",
            "test_type": "Technical",
            "description": "Assessment for HTML5 skills."
        },
        {
            "title": "CSS3 (New)",
            "url": "https://www.shl.com/solutions/products/product-catalog/view/css3-new/",
            "remote_testing_support": "Yes",
            "adaptive_irt_support": "No",
            "duration": "30 minutes",
            "test_type": "Technical",
            "description": "Assessment for CSS3 skills."
        },
        {
            "title": "Selenium (New)",
            "url": "https://www.shl.com/solutions/products/product-catalog/view/selenium-new/",
            "remote_testing_support": "Yes",
            "adaptive_irt_support": "No",
            "duration": "45 minutes",
            "test_type": "Technical",
            "description": "Assessment for Selenium automation testing skills."
        },
        {
            "title": "SQL Server (New)",
            "url": "https://www.shl.com/solutions/products/product-catalog/view/sql-server-new/",
            "remote_testing_support": "Yes",
            "adaptive_irt_support": "No",
            "duration": "40 minutes",
            "test_type": "Technical",
            "description": "Assessment for SQL Server database skills."
        },
        {
            "title": "Automata - SQL (New)",
            "url": "https://www.shl.com/solutions/products/product-catalog/view/automata-sql-new/",
            "remote_testing_support": "Yes",
            "adaptive_irt_support": "No",
            "duration": "60 minutes",
            "test_type": "Technical",
            "description": "Hands-on SQL coding assessment."
        },
        {
            "title": "Manual Testing (New)",
            "url": "https://www.shl.com/solutions/products/product-catalog/view/manual-testing-new/",
            "remote_testing_support": "Yes",
            "adaptive_irt_support": "No",
            "duration": "45 minutes",
            "test_type": "Technical",
            "description": "Assessment for manual testing skills."
        },
        {
            "title": "Verify - Numerical Ability",
            "url": "https://www.shl.com/solutions/products/product-catalog/view/verify-numerical-ability/",
            "remote_testing_support": "Yes",
            "adaptive_irt_support": "Yes",
            "duration": "25 minutes",
            "test_type": "Cognitive",
            "description": "Assessment for numerical reasoning skills."
        },
        {
            "title": "Financial Professional - Short Form",
            "url": "https://www.shl.com/solutions/products/product-catalog/view/financial-professional-short-form/",
            "remote_testing_support": "Yes",
            "adaptive_irt_support": "No",
            "duration": "30 minutes",
            "test_type": "Skills",
            "description": "Assessment for financial professional skills."
        },
        {
            "title": "Bank Administrative Assistant - Short Form",
            "url": "https://www.shl.com/solutions/products/product-catalog/view/bank-administrative-assistant-short-form/",
            "remote_testing_support": "Yes",
            "adaptive_irt_support": "No",
            "duration": "35 minutes",
            "test_type": "Skills",
            "description": "Assessment for bank administrative assistant skills."
        },
        {
            "title": "SHL Verify Interactive - Inductive Reasoning",
            "url": "https://www.shl.com/solutions/products/product-catalog/view/shl-verify-interactive-inductive-reasoning/",
            "remote_testing_support": "Yes",
            "adaptive_irt_support": "Yes",
            "duration": "25 minutes",
            "test_type": "Cognitive",
            "description": "Interactive assessment for inductive reasoning skills."
        }
    ]
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Save to JSON
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(assessments, f, indent=4)
        logger.info(f"Saved {len(assessments)} mock assessments to {output_file}")
        return assessments
    except Exception as e:
        logger.error(f"Error saving mock data to JSON: {e}")
        return []

if __name__ == "__main__":
    scraper = SHLCatalogScraper()
    
    # Try to load existing data first
    if not scraper.load_from_json():
        # Try scraping first
        try:
            scraper.scrape_catalog_pages()
            
            # If we didn't get any assessments, use mock data
            if len(scraper.assessments) == 0:
                logger.warning("No assessments scraped, generating mock data")
                scraper.assessments = generate_mock_data()
        except Exception as e:
            logger.error(f"Scraping failed: {e}, using mock data instead")
            scraper.assessments = generate_mock_data()