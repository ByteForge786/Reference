import pandas as pd
import time
import logging
from typing import Dict, Tuple, List
import json
from datetime import datetime
from test import chinou_response

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pii_processor.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class AdaptiveRateLimiter:
    def __init__(self, initial_delay: float = 1.0, max_delay: float = 60.0, backoff_factor: float = 2.0):
        self.current_delay = initial_delay
        self.max_delay = max_delay
        self.backoff_factor = backoff_factor
        self.success_count = 0
        self.required_successes = 5  # Number of successful calls before reducing delay
        
    def increase_delay(self):
        """Increase delay after encountering an error"""
        self.current_delay = min(self.current_delay * self.backoff_factor, self.max_delay)
        self.success_count = 0
        logger.info(f"Increased delay to {self.current_delay} seconds")
        
    def decrease_delay(self):
        """Potentially decrease delay after successful calls"""
        self.success_count += 1
        if self.success_count >= self.required_successes:
            self.current_delay = max(self.current_delay / self.backoff_factor, 1.0)
            self.success_count = 0
            logger.info(f"Decreased delay to {self.current_delay} seconds")
            
    def wait(self):
        """Wait for the current delay period"""
        time.sleep(self.current_delay)

def create_llm_prompt(attr_name: str, description: str) -> str:
    """Create a prompt for the LLM"""
    return f"""Given the following attribute name and description, please:
1. Enhance the description to be more detailed and clear (in two lines)
2. Classify it as either 'Sensitive PII' or 'Non-Sensitive PII'

Attribute Name: {attr_name}
Description: {description}

Please return the response in the following JSON format:
{{
    "enhanced_description": "first line\\nsecond line",
    "classification": "Sensitive PII or Non-Sensitive PII"
}}"""

def parse_llm_response(response: str) -> Dict:
    """Parse the LLM response and handle potential errors"""
    try:
        return json.loads(response)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse LLM response: {response}")
        logger.error(f"Error: {str(e)}")
        # Return a default structure
        return {
            "enhanced_description": "Error processing description\nPlease review manually",
            "classification": "Non-Sensitive PII"
        }

def process_attributes(input_file: str, output_file: str):
    """Main function to process the CSV file"""
    try:
        # Read input CSV
        df = pd.read_csv(input_file)
        logger.info(f"Successfully loaded {len(df)} rows from {input_file}")
        
        # Initialize rate limiter
        rate_limiter = AdaptiveRateLimiter()
        
        # Initialize result lists
        enhanced_descriptions = []
        classifications = []
        
        # Process each row
        for idx, row in df.iterrows():
            attr_name = row['attribute_name']
            description = row['description']
            
            success = False
            attempts = 0
            max_attempts = 3
            
            while not success and attempts < max_attempts:
                try:
                    # Wait according to rate limiter
                    rate_limiter.wait()
                    
                    # Create and send prompt to LLM
                    prompt = create_llm_prompt(attr_name, description)
                    response = chinou_response(prompt)
                    
                    # Parse response
                    parsed_response = parse_llm_response(response)
                    
                    # Store results
                    enhanced_descriptions.append(parsed_response['enhanced_description'])
                    classifications.append(parsed_response['classification'])
                    
                    # Update rate limiter on success
                    rate_limiter.decrease_delay()
                    success = True
                    
                    logger.info(f"Successfully processed attribute: {attr_name}")
                    
                except Exception as e:
                    attempts += 1
                    rate_limiter.increase_delay()
                    logger.warning(f"Attempt {attempts} failed for {attr_name}: {str(e)}")
                    
                    if attempts == max_attempts:
                        logger.error(f"Failed to process {attr_name} after {max_attempts} attempts")
                        enhanced_descriptions.append("Processing failed\nPlease review manually")
                        classifications.append("Non-Sensitive PII")
        
        # Create output DataFrame
        output_df = df.copy()
        output_df['enhanced_description'] = enhanced_descriptions
        output_df['pii_classification'] = classifications
        
        # Save to CSV
        output_df.to_csv(output_file, index=False)
        logger.info(f"Successfully saved results to {output_file}")
        
    except Exception as e:
        logger.error(f"Critical error in process_attributes: {str(e)}")
        raise

if __name__ == "__main__":
    input_file = "attributes.csv"
    output_file = f"processed_attributes_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    
    try:
        process_attributes(input_file, output_file)
    except Exception as e:
        logger.error(f"Script failed: {str(e)}")
