import pandas as pd
import threading
from queue import Queue
import time
import logging
from concurrent.futures import ThreadPoolExecutor
import backoff
import json
from typing import Dict, List, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('attribute_processing.log'),
        logging.StreamHandler()
    ]
)

class AdaptiveRateLimiter:
    def __init__(self, initial_rate: float = 10.0):
        self.current_rate = initial_rate
        self.window_size = 1.0  # 1 second
        self.lock = threading.Lock()
        self.last_success = time.time()
        self.consecutive_failures = 0

    def update_rate(self, success: bool):
        with self.lock:
            if success:
                self.consecutive_failures = 0
                self.current_rate = min(self.current_rate * 1.1, 50)
            else:
                self.consecutive_failures += 1
                self.current_rate = max(self.current_rate * 0.5, 1)
            
            logging.info(f"Rate limit adjusted to {self.current_rate:.2f} requests/second")

    def wait(self):
        with self.lock:
            current_time = time.time()
            time_since_last = current_time - self.last_success
            wait_time = max(0, (1 / self.current_rate) - time_since_last)
            if wait_time > 0:
                time.sleep(wait_time)
            self.last_success = time.time()

def clean_string(s: str) -> str:
    """Clean string by removing spaces and converting to lowercase."""
    return str(s).lower().replace(" ", "")

def generate_llm_prompt(row: Dict) -> str:
    """Generate a prompt for the LLM based on the attribute metadata."""
    
    # Clean and standardize input strings
    source = clean_string(row['sourced_from'])
    sensitivity = clean_string(row['sensitivity'])
    
    # Check if attribute is user-related (either HR data or non-sensitive PII)
    is_hr_data = any(src in source for src in ['hr', 'successfactor'])
    is_nonsensitive_pii = 'nonsensitivepii' in sensitivity
    is_user_related = is_hr_data or is_nonsensitive_pii
    
    base_prompt = f"""As a data dictionary expert at Nomura Holdings, provide a comprehensive description for the following attribute:

Attribute Name: {row['attribute_name']}
Source System: {row['sourced_from']}
Data Sensitivity: {row['sensitivity']}

Context:
- This is a {'user-related' if is_user_related else 'business-related'} attribute
- Nomura Holdings is a global financial services company
- {'This attribute contains personal information' if is_hr_data else 'This attribute relates to internal business operations'}

Provide a single comprehensive description that explains both what this attribute represents and its business purpose.
Keep the description under 150 characters.

Return the response in the following JSON format:
{{
    "attribute_name": "{row['attribute_name']}",
    "description": "your description here"
}}
"""
    return base_prompt

@backoff.on_exception(
    backoff.expo,
    Exception,
    max_tries=5,
    max_time=300
)
def call_llm_with_retry(prompt: str, rate_limiter: AdaptiveRateLimiter) -> Dict:
    """Call the LLM API with retry logic and adaptive rate limiting."""
    try:
        rate_limiter.wait()
        response = chinou_response(prompt)  # Your actual API call
        
        # Ensure we have valid JSON
        if isinstance(response, str):
            parsed_response = json.loads(response)
        else:
            parsed_response = response
            
        rate_limiter.update_rate(True)
        return parsed_response
    except json.JSONDecodeError as e:
        logging.error(f"Failed to parse JSON response: {response}")
        rate_limiter.update_rate(False)
        raise
    except Exception as e:
        rate_limiter.update_rate(False)
        logging.error(f"API call failed: {str(e)}")
        raise

def process_batch(batch: pd.DataFrame, rate_limiter: AdaptiveRateLimiter) -> List[Tuple[str, str]]:
    """Process a batch of attributes."""
    results = []
    for _, row in batch.iterrows():
        try:
            prompt = generate_llm_prompt(row)
            response = call_llm_with_retry(prompt, rate_limiter)
            
            # Extract description from JSON response
            description = response.get('description', 'Failed to generate description')
            results.append((row['attribute_name'], description))
            
            logging.info(f"Successfully processed attribute: {row['attribute_name']}")
        except Exception as e:
            logging.error(f"Failed to process attribute {row['attribute_name']}: {str(e)}")
            results.append((row['attribute_name'], 'Failed to generate description'))
    return results

def main():
    # Read input CSV
    input_file = 'attributes.csv'
    output_file = 'attributes_with_descriptions.csv'
    batch_size = 50
    max_workers = 4

    logging.info(f"Starting processing of {input_file}")
    
    try:
        df = pd.read_csv(input_file)
        total_records = len(df)
        logging.info(f"Loaded {total_records} attributes for processing")

        # Initialize rate limiter
        rate_limiter = AdaptiveRateLimiter()

        # Split into batches
        batches = [df[i:i + batch_size] for i in range(0, len(df), batch_size)]
        results = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(process_batch, batch, rate_limiter)
                for batch in batches
            ]

            for future in futures:
                batch_results = future.result()
                results.extend(batch_results)

        # Update DataFrame with results
        result_dict = dict(results)
        df['description'] = df['attribute_name'].map(result_dict)

        # Save results
        df.to_csv(output_file, index=False)
        logging.info(f"Processing completed. Results saved to {output_file}")

    except Exception as e:
        logging.error(f"Main processing failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
