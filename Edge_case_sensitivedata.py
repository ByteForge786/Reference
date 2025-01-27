import json
import csv
import time
import logging
from typing import List, Dict
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'pii_generator_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Rate limiting parameters
MIN_DELAY = 1  # Minimum delay between requests in seconds
MAX_DELAY = 16  # Maximum delay in case of throttling
BATCH_SIZE = 10  # Number of pairs to request in each batch

class AdaptiveRateLimiter:
    def __init__(self, initial_delay: float = MIN_DELAY):
        self.current_delay = initial_delay
        self.success_count = 0
        
    def wait(self):
        time.sleep(self.current_delay)
    
    def success(self):
        self.success_count += 1
        if self.success_count >= 3 and self.current_delay > MIN_DELAY:
            self.current_delay = max(MIN_DELAY, self.current_delay / 2)
            self.success_count = 0
            logger.info(f"Decreased delay to {self.current_delay:.2f} seconds")
    
    def failure(self):
        self.current_delay = min(MAX_DELAY, self.current_delay * 2)
        self.success_count = 0
        logger.warning(f"Increased delay to {self.current_delay:.2f} seconds")

def create_llm_prompt(batch_size: int) -> str:
    prompt = """Task: Generate attribute edge cases where the classification between Sensitive PII and Non-sensitive PII depends on the content detail level and included information.

Context:
Sensitive PII includes data that could harm individuals if exposed, such as:
- Government-issued IDs (SSN, passport, driver's license)
- Financial data (account numbers, PINs, credit cards)
- Biometric data (fingerprints, DNA, facial geometry)
- Health information (medical records, prescriptions)
- Precise geolocation
- Religious beliefs
- Racial/ethnic origin
- Trade union membership
- Sexual orientation
- Criminal history
- Private communications content
- Account credentials with passwords
- Mental health data

Non-sensitive PII includes basic identifying information that doesn't create risk of harm if exposed, such as:
- Names
- Business contact information
- Job titles
- Office locations
- Work schedules
- General preferences
- Device settings
- Basic demographics

Format:
For each attribute, create two versions:
1. Basic version (Non-sensitive PII): Contains general information without personal identifiers or sensitive details
2. Detailed version (Sensitive PII): Same attribute enhanced with sensitive elements that change its classification

Please provide the response in JSON format with the following structure for each attribute:
{
    "attributes": [
        {
            "attribute_name": "string",
            "description": "string",
            "label": "string"
        }
    ]
}

Description requirements:
- Start with "User's" or "Person's" or "Individual's"
- Provide one comprehensive sentence explaining the content and purpose
- Don't use words like "sensitive" or "non-sensitive"
- Include specific examples of included data elements
- Focus on practical, real-world usage

Example pairs:
"investment_portfolio_basic","User's publicly traded stock holdings and market investment preferences tracked for portfolio analysis","Non-sensitive PII"
"investment_portfolio_detailed","User's stock holdings combined with account PINs, trading passwords, and bank account numbers used for automated trading","Sensitive PII"

Generate {batch_size} pairs of attributes (basic and detailed versions) covering various business contexts.
"""
    return prompt.format(batch_size=batch_size)

def generate_pii_attributes(total_pairs: int) -> List[Dict]:
    rate_limiter = AdaptiveRateLimiter()
    all_attributes = []
    batches = (total_pairs + BATCH_SIZE - 1) // BATCH_SIZE

    for batch in range(batches):
        remaining_pairs = min(BATCH_SIZE, total_pairs - batch * BATCH_SIZE)
        logger.info(f"Processing batch {batch + 1}/{batches} ({remaining_pairs} pairs)")
        
        try:
            rate_limiter.wait()
            
            # Create prompt for current batch
            prompt = create_llm_prompt(remaining_pairs)
            
            # Make API call
            response = chinou_response(prompt)  # This is your LLM API function
            
            # Parse JSON response
            try:
                attributes = json.loads(response)["attributes"]
                logger.info(f"Successfully generated {len(attributes)} attributes")
                all_attributes.extend(attributes)
                rate_limiter.success()
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON response: {e}")
                rate_limiter.failure()
                continue
                
        except Exception as e:
            logger.error(f"Error in batch {batch + 1}: {e}")
            rate_limiter.failure()
            continue

    return all_attributes

def save_to_csv(attributes: List[Dict], filename: str):
    try:
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['attribute_name', 'description', 'label'])
            writer.writeheader()
            writer.writerows(attributes)
        logger.info(f"Successfully saved {len(attributes)} attributes to {filename}")
    except Exception as e:
        logger.error(f"Failed to save CSV file: {e}")
        raise

def main():
    try:
        logger.info("Starting PII attribute generation")
        
        # Generate 60 pairs (120 total attributes)
        total_pairs = 60
        attributes = generate_pii_attributes(total_pairs)
        
        # Save results
        output_file = f'pii_attributes_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        save_to_csv(attributes, output_file)
        
        logger.info("PII attribute generation completed successfully")
        
    except Exception as e:
        logger.error(f"Process failed: {e}")
        raise

if __name__ == "__main__":
    main()
