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
