def process_attributes(input_file: str, output_file: str):
    """Main function to process the CSV file"""
    try:
        # Read input CSV
        df = pd.read_csv(input_file)
        logger.info(f"Successfully loaded {len(df)} rows from {input_file}")
        
        # Filter rows based on label
        valid_labels = ['Sensitive PII', 'Non-Sensitive PII']
        mask = df['label'].isin(valid_labels)
        filtered_df = df[mask]
        logger.info(f"Filtered to {len(filtered_df)} rows with valid PII labels")
        
        # Initialize rate limiter
        rate_limiter = AdaptiveRateLimiter()
        
        # Initialize result lists
        enhanced_descriptions = []
        classifications = []
        
        # Process each filtered row
        for idx, row in filtered_df.iterrows():
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
        # Start with original dataframe
        output_df = df.copy()
        
        # Initialize new columns with original values
        output_df['enhanced_description'] = df['description']  # Copy original descriptions
        output_df['pii_classification'] = df['label']  # Copy original labels
        
        # Update only the filtered rows with their new values
        output_df.loc[filtered_df.index, 'enhanced_description'] = enhanced_descriptions
        output_df.loc[filtered_df.index, 'pii_classification'] = classifications
        
        # Save to CSV
        output_df.to_csv(output_file, index=False)
        logger.info(f"Successfully saved results to {output_file}")
        
    except Exception as e:
        logger.error(f"Critical error in process_attributes: {str(e)}")
        raise







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
    prompt_parts.append("- Nomura Holdings is a global financial services company")
    prompt_parts.append(f"- {'This attribute contains personal information' if is_user_related else 'This attribute relates to internal business operations'}")

    prompt_parts.append(f"""
Return the response in the following JSON format:
{{
    "attribute_name": "{row['attribute_name']}",
    "description": "your description here"
}}
""")
    
    return "\n".join(prompt_parts)
