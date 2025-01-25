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

def clean_string(s) -> str:
    """Clean string by removing spaces and converting to lowercase. Handle None/NaN values."""
    if pd.isna(s) or s is None:
        return ""
    return str(s).lower().replace(" ", "")

def safe_get_column(df: pd.DataFrame, column: str) -> pd.Series:
    """Safely get a column from DataFrame, return empty string if column doesn't exist."""
    return df[column] if column in df.columns else pd.Series([""] * len(df))

def determine_user_relation(row: Dict, available_columns: set) -> Tuple[bool, List[str]]:
    """Determine if attribute is user-related and collect context reasons."""
    reasons = []
    
    # Check source system if available
    if 'sourced_from' in available_columns:
        source = clean_string(row.get('sourced_from', ''))
        if any(src in source for src in ['hr', 'successfactor']):
            reasons.append("HR system source")
    
    # Check sensitivity if available
    if 'sensitivity' in available_columns:
        sensitivity = clean_string(row.get('sensitivity', ''))
        if 'nonsensitivepii' in sensitivity:
            reasons.append("Contains non-sensitive PII")
    
    is_user_related = len(reasons) > 0
    return is_user_related, reasons

def generate_llm_prompt(row: Dict) -> str:
    """Generate a prompt for the LLM based on the attribute metadata."""
    
    # Get available columns
    available_columns = set(row.keys())
    
    # Determine user relation and get context
    is_user_related, reasons = determine_user_relation(row, available_columns)
    
    # Build context information
    context_info = []
    if reasons:  # If we have specific reasons for user-relation
        context_info.extend(reasons)
    elif is_user_related:
        context_info.append("User-related attribute")
    else:
        context_info.append("Business-related attribute")
    
    # Build the prompt with available information
    prompt_parts = ["As a data dictionary expert at Nomura Holdings, provide a comprehensive description for the following attribute:"]
    
    # Always include attribute name
    prompt_parts.append(f"Attribute Name: {row['attribute_name']}")
    
    # Add optional fields if available
    if 'sourced_from' in available_columns and not pd.isna(row['sourced_from']):
        prompt_parts.append(f"Source System: {row['sourced_from']}")
    if 'sensitivity' in available_columns and not pd.isna(row['sensitivity']):
        prompt_parts.append(f"Data Sensitivity: {row['sensitivity']}")
    
    # Add context section
    prompt_parts.append("\nContext:")
    for info in context_info:
        prompt_parts.append(f"- {info}")
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
        # Read CSV and handle missing columns
        df = pd.read_csv(input_file)
        required_columns = {'attribute_name'}
        optional_columns = {'sourced_from', 'sensitivity'}
        
        # Validate required columns
        missing_required = required_columns - set(df.columns)
        if missing_required:
            raise ValueError(f"Missing required columns: {missing_required}")
            
        # Log column availability
        available_optional = optional_columns.intersection(set(df.columns))
        missing_optional = optional_columns - available_optional
        if missing_optional:
            logging.warning(f"Optional columns not found: {missing_optional}")
        
        # Fill missing optional columns with empty strings
        for col in optional_columns:
            if col not in df.columns:
                df[col] = ""
        
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
