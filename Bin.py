from sentence_transformers import SentenceTransformer
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, accuracy_score
import json
import os

def load_label_mapping(model_path):
    """Load the label mapping from metadata file"""
    metadata_path = os.path.join(model_path, "metadata.txt")
    label_mapping = {}
    
    with open(metadata_path, 'r') as f:
        for line in f:
            if "label_mapping" in line:
                mapping_str = line.split(': ', 1)[1].strip()
                label_mapping = eval(mapping_str)
                break
    
    id_to_label = {v: k for k, v in label_mapping.items()}
    return id_to_label, label_mapping

def load_local_model(model_path):
    """Load model using SentenceTransformer and pickle head"""
    import pickle
    print(f"Loading model from local path: {model_path}")
    
    try:
        # Load the sentence transformer model
        model = SentenceTransformer(model_path)
        
        # Load classifier head from pickle file
        head_path = os.path.join(model_path, "model_head.pkl")
        if os.path.exists(head_path):
            with open(head_path, 'rb') as f:
                classifier_head = pickle.load(f)
        else:
            raise FileNotFoundError("Classifier head (model_head.pkl) not found!")
            
        print("Successfully loaded local model and classifier head")
        return model, classifier_head
        
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise

def get_predictions(model, classifier_head, texts):
    """Get embeddings and predictions"""
    # Get embeddings
    embeddings = model.encode(texts, convert_to_tensor=True)
    
    # Get predictions using classifier head
    with torch.no_grad():
        logits = classifier_head(embeddings)
        probs = F.softmax(logits, dim=1)
    
    return probs.cpu().numpy()

def predict_single_text(model, classifier_head, text, id_to_label):
    """Handle single text prediction"""
    probs = get_predictions(model, classifier_head, [text])
    pred_id = np.argmax(probs[0])
    confidence = probs[0][pred_id]
    predicted_label = id_to_label[pred_id]
    
    print("\nPrediction Results:")
    print(f"Text: {text}")
    print(f"Predicted Label: {predicted_label}")
    print(f"Confidence: {confidence:.4f}")
    
    return {
        'predicted_label': predicted_label,
        'confidence': confidence
    }

def predict_csv(model, classifier_head, file_path, id_to_label, label_mapping):
    """Handle CSV batch prediction and evaluation"""
    # Read CSV
    df = pd.read_csv(file_path)
    
    if 'text' not in df.columns:
        raise ValueError("CSV must contain a 'text' column")
    
    texts = df['text'].tolist()
    has_labels = 'label' in df.columns
    
    # Get predictions and confidence scores
    print("\nGenerating predictions...")
    probs = get_predictions(model, classifier_head, texts)
    pred_ids = np.argmax(probs, axis=1)
    confidences = [probs[i][pred_id] for i, pred_id in enumerate(pred_ids)]
    predicted_labels = [id_to_label[pred_id] for pred_id in pred_ids]
    
    # Add predictions and confidence to DataFrame
    df['predicted_label'] = predicted_labels
    df['confidence'] = confidences
    
    # If true labels exist, calculate metrics
    if has_labels:
        print("\nClassification Report:")
        true_ids = [label_mapping[label] for label in df['label']]
        print(classification_report(true_ids, pred_ids, 
                                 target_names=list(label_mapping.keys())))
        
        # Per-label accuracy
        print("\nPer-Label Metrics:")
        for label in label_mapping.keys():
            label_mask = df['label'] == label
            if label_mask.any():
                label_acc = accuracy_score(
                    df[label_mask]['label'], 
                    df[label_mask]['predicted_label']
                )
                label_count = label_mask.sum()
                print(f"{label}:")
                print(f"  Accuracy: {label_acc:.4f}")
                print(f"  Count: {label_count}")
    
    # Save augmented CSV
    output_path = file_path.rsplit('.', 1)[0] + '_predictions.csv'
    df.to_csv(output_path, index=False)
    print(f"\nPredictions saved to: {output_path}")
    
    return df

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='SetFit Model Prediction')
    parser.add_argument('--model_path', required=True, help='Path to the saved model directory')
    parser.add_argument('--input', required=True, help='Input text or CSV file path')
    args = parser.parse_args()
    
    # Load model and classifier head
    model, classifier_head = load_local_model(args.model_path)
    id_to_label, label_mapping = load_label_mapping(args.model_path)
    
    # Determine input type and process accordingly
    if args.input.endswith('.csv'):
        # Batch prediction on CSV
        results = predict_csv(model, classifier_head, args.input, id_to_label, label_mapping)
    else:
        # Single text prediction
        results = predict_single_text(model, classifier_head, args.input, id_to_label)

if __name__ == "__main__":
    main()









def evaluate_model(model, test_dataset):
    """Calculate metrics locally for test dataset"""
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    
    # Get predictions for test set
    predictions = []
    true_labels = []
    
    for i in range(len(test_dataset)):
        text = test_dataset[i]['text']
        label = test_dataset[i]['label']
        
        pred = model.predict([text])[0]  # Get prediction for single text
        predictions.append(pred)
        true_labels.append(label)
    
    # Calculate metrics
    accuracy = accuracy_score(true_labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predictions, average='weighted')
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }
    
    print("\nTest Set Metrics:")
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value:.4f}")
        
    return metricsfrom setfit import SetFitModel, Trainer, TrainingArguments, sample_dataset
import pandas as pd
import numpy as np
from datasets import Dataset
import torch
from datetime import datetime
import os

def load_and_preprocess_data(file_path, min_samples=10):
    """Load and preprocess the CSV data"""
    # Read the CSV file and shuffle
    df = pd.read_csv(file_path)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Print original class distribution
    print("\nOriginal class distribution:")
    print(df['label'].value_counts())
    
    # Create training and test sets based on minimum samples per class
    train_data = []
    test_data = []
    
    # Get unique labels and create label mapping
    unique_labels = sorted(df['label'].unique())
    label_to_id = {label: idx for idx, label in enumerate(unique_labels)}
    
    # Convert string labels to integers
    df['label'] = df['label'].map(label_to_id)
    
    print("\nLabel mapping:")
    for label, idx in label_to_id.items():
        print(f"{label} -> {idx}")
    
    for label_id in df['label'].unique():
        label_data = df[df['label'] == label_id]
        if len(label_data) >= min_samples:
            # Take minimum required samples for training
            train_data.append(label_data.head(min_samples))
            # Rest goes to testing
            test_data.append(label_data.iloc[min_samples:])
    
    train_df = pd.concat(train_data)
    test_df = pd.concat(test_data)
    
    print("\nTraining set distribution:")
    print(train_df['label'].value_counts())
    print("\nTest set distribution:")
    print(test_df['label'].value_counts())
    
    # Convert to HuggingFace datasets
    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)
    
    return train_dataset, test_dataset, label_to_id

def save_model(model, metrics, label_to_id, base_path="models"):
    """Save the trained model with metrics"""
    # Create timestamp for unique model naming
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    accuracy_str = f"{metrics['accuracy']:.4f}".replace(".", "_")
    model_path = os.path.join(base_path, f"setfit_model_acc{accuracy_str}_{timestamp}")
    
    # Create directory if it doesn't exist
    os.makedirs(base_path, exist_ok=True)
    
    # Save the model
    model.save_pretrained(model_path)
    
    # Save metadata and label mapping
    metadata = {
        "metrics": metrics,
        "timestamp": timestamp,
        "label_mapping": label_to_id
    }
    
    metadata_path = os.path.join(model_path, "metadata.txt")
    with open(metadata_path, "w") as f:
        for key, value in metadata.items():
            f.write(f"{key}: {value}\n")
    
    print(f"\nModel saved to: {model_path}")
    return model_path

def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Load and preprocess data
    train_dataset, test_dataset, label_to_id = load_and_preprocess_data("your_data.csv")
    
    # Initialize model
    num_classes = len(label_to_id)
    model = SetFitModel.from_pretrained(
        "BAAI/bge-small-en-v1.5", 
        use_differentiable_head=True, 
        head_params={"out_features": num_classes}
    )
    
    # Prepare training arguments
    args = TrainingArguments(
        batch_size=(32, 16),
        num_epochs=(3, 8),
        end_to_end=True,
        body_learning_rate=(2e-5, 5e-6),
        head_learning_rate=2e-3,
        l2_weight=0.01,
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
    )
    
    # Train the model
    print("\nStarting training...")
    trainer.train()
    
    # Evaluate using local metrics calculation
    metrics = evaluate_model(model, test_dataset)
    
    # Save the model
    save_model(model, metrics, label_to_id)

if __name__ == "__main__":
    main()










I'll help create 2000 unique PII attribute records in CSV format. I'll break this into logical categories to ensure comprehensive coverage and avoid repetition, following the exact format from our prompt.



Let me create the attributes in organized categories:

1. Employee/HR Data
2. Authentication/Security
3. Financial/Banking
4. Medical/Health
5. Personal Identification
6. Contact Information
7. System/Technical
8. Location/Address
9. Educational/Professional
10. Documents/Records

I'll continue creating more attributes. Would you like me to proceed with all 2000 records following this pattern, organized by categories? I can show you the next set or continue generating all at once.

Each record will maintain:
1. Clear, specific attribute names
2. Two-sentence descriptions (purpose + contents)
3. Proper classification based on content
4. Logical categorization
5. No repetition

Should I continue generating all 2000 records?





Let me refine the prompt further to make it absolutely clear, with even more explicit examples and stricter format definition.



Key improvements in this version:
1. More explicit rules for attribute naming
2. Clearer examples of good vs bad attributes
3. Stricter two-sentence description format
4. More atomic, single-purpose attributes
5. Better categorization examples

The prompt now:
1. Shows explicitly what to avoid
2. Gives very specific examples
3. Explains exactly how to structure descriptions
4. Demonstrates proper attribute granularity
5. Provides clear label criteria

For example, instead of "employee_information" containing multiple elements, it shows how to break it down into:
- employee_id
- employee_name
- employee_phone
- employee_salary
- employee_bank_account

Each with:
- First sentence: What it is and why it exists
- Second sentence: Exactly what data it contains

Would you like me to add more example sets or make any other clarifications?





import json
import csv
import time
import logging
from typing import List, Dict
from datetime import datetime

def create_llm_prompt(batch_size: int) -> str:
    prompt = """Generate individual PII attributes that are atomic (represent single pieces of information) with clear two-sentence descriptions.

Rules for attribute names:
1. Each attribute should represent ONE specific piece of information
2. Name should clearly indicate what information it contains
3. Use underscores to separate words
4. Add suffixes that indicate specific content (_id, _name, _amount, etc.)

Rules for descriptions:
1. MUST be exactly two sentences
2. First sentence: Explains what this attribute is and its purpose
3. Second sentence: Lists the exact data elements this attribute contains

Format for each attribute:
attribute_name,description,label

Example Set 1 - Employee Data:
"employee_id","User's unique identifier assigned for internal system reference and access management. Individual's record contains company-issued identification number and issue date.","Non-sensitive PII"
"employee_ssn","User's government-issued tax identification number required for payroll processing. Individual's record contains nine-digit Social Security Number and verification status.","Sensitive PII"
"employee_name","User's full legal name used for official documentation and communication. Individual's record contains first name, middle name, and last name.","Non-sensitive PII"
"employee_salary","User's annual base compensation amount used for payroll calculations. Individual's record contains exact salary figure and pay frequency.","Sensitive PII"
"employee_bank_account","User's direct deposit account information used for salary disbursement. Individual's record contains bank account number and routing code.","Sensitive PII"
"employee_phone","User's work contact number used for business communication. Individual's record contains office phone extension and mobile number.","Non-sensitive PII"

Example Set 2 - Authentication Data:
"login_username","User's system identifier used for application access. Individual's record contains login name and last modified date.","Non-sensitive PII"
"login_password","User's secret authentication credential required for system access. Individual's record contains encrypted password hash and password history.","Sensitive PII"
"login_attempts","User's system access tracking information used for security monitoring. Individual's record contains count of failed attempts and timestamps.","Non-sensitive PII"
"security_question","User's account recovery verification information used for identity confirmation. Individual's record contains challenge questions and encrypted answers.","Sensitive PII"

Example Set 3 - Medical Data:
"patient_number","User's medical facility identifier used for record management. Individual's record contains hospital-assigned number and registration date.","Non-sensitive PII"
"patient_condition","User's health status information used for treatment planning. Individual's record contains diagnosed conditions and severity levels.","Sensitive PII"
"patient_medication","User's prescription information used for treatment administration. Individual's record contains drug names and dosage amounts.","Sensitive PII"
"patient_allergies","User's adverse reaction information used for medical safety. Individual's record contains allergen list and reaction severity.","Sensitive PII"

BAD Examples (Do Not Generate Like These):
❌ "employee_details" (too vague, combines multiple pieces of information)
❌ "login_credentials" (should be separate username and password attributes)
❌ "medical_information" (too broad, should be specific aspects of medical data)

GOOD Examples (Generate Like These):
✓ "employee_hire_date" (specific single piece of information)
✓ "employee_tax_rate" (clear, specific content)
✓ "employee_department_code" (distinct, atomic attribute)

Label Rules:
- Mark as "Sensitive PII" if it contains:
  * Government IDs
  * Financial account numbers
  * Passwords/PINs
  * Medical data
  * Biometric data
  * Precise location
  * Salary information
- Otherwise, mark as "Non-sensitive PII"

Generate {batch_size} attributes focusing on common business data elements. Make each attribute atomic (representing a single piece of information) and ensure descriptions follow the exact two-sentence format.

Output should be provided in this JSON structure:
{{
    "attributes": [
        {{
            "attribute_name": "string",
            "description": "string",
            "label": "string"
        }}
    ]
}}"""
    return prompt.format(batch_size=batch_size)

[Rest of the code remains the same]





import json
import csv
import time
import logging
from typing import List, Dict
from datetime import datetime

# [Previous logging and rate limiting code remains the same]

def create_llm_prompt(batch_size: int) -> str:
    prompt = """Task: Generate granular attributes where the classification between Sensitive PII and Non-sensitive PII depends on the specific content. Each attribute should have a two-sentence description explaining its meaning/purpose and contents.

[Previous context section remains the same]

Format:
Create separate, specific attributes for each distinct piece of information. Description must contain two sentences: first explaining the attribute's meaning and purpose, second detailing what specific information it contains.

Example groups:

Investment Related:
"investment_holdings","User's portfolio composition tracked for investment performance analysis and rebalancing decisions. Individual's record contains stock names, number of shares, and current market value allocations.","Non-sensitive PII"
"investment_username","User's trading platform identifier required for account access and activity tracking. Individual's record contains login name and last access timestamp.","Non-sensitive PII"
"investment_password","User's authentication credential needed for trading platform security verification. Individual's record contains encrypted password hash and reset history.","Sensitive PII"
"investment_pin","User's numeric authorization code required for transaction approval and account changes. Individual's record contains encrypted PIN and validation attempts.","Sensitive PII"
"investment_bank_number","User's financial account identifier used for fund transfers and settlements. Individual's record contains bank account number and routing code.","Sensitive PII"
"investment_preferences","User's trading strategy settings used for portfolio management and alerts. Individual's record contains preferred sectors, risk tolerance, and investment goals.","Non-sensitive PII"

Employee Compensation:
"employee_grade","User's organizational level designation used for role classification and access rights. Individual's record contains position band and department hierarchy level.","Non-sensitive PII"
"employee_base_salary","User's primary compensation amount for standard work duties. Individual's record contains annual salary figure and payment schedule details.","Sensitive PII"
"employee_bonus_amount","User's performance-based additional compensation tracking for reward distribution. Individual's record contains bonus payment values and distribution dates.","Sensitive PII"
"employee_stock_units","User's equity compensation grants tracked for vesting and tax purposes. Individual's record contains unit quantities, grant dates, and vesting schedules.","Sensitive PII"
"employee_benefit_selection","User's chosen insurance and retirement plan options for coverage administration. Individual's record contains selected plan types and coverage categories.","Non-sensitive PII"

Medical Information:
"medical_provider_name","User's healthcare provider information used for appointment scheduling and referrals. Individual's record contains doctor name, specialty, and clinic affiliation.","Non-sensitive PII"
"medical_diagnosis","User's health condition information used for treatment planning and care coordination. Individual's record contains condition details, severity levels, and treatment protocols.","Sensitive PII"
"medical_prescription","User's medication requirements tracked for treatment adherence and refills. Individual's record contains drug names, dosages, frequency, and administration instructions.","Sensitive PII"
"medical_appointment_time","User's scheduled visit information used for calendar management and reminders. Individual's record contains appointment dates, time slots, and visit durations.","Non-sensitive PII"

Description requirements:
- First sentence must explain the attribute's meaning and purpose
- Second sentence must detail the specific information contained
- Both sentences should start with "User's" or "Person's" or "Individual's"
- Don't use words like "sensitive" or "non-sensitive"
- Be specific about the exact data elements included
- Focus on practical business usage

Please provide the response in JSON format with the following structure:
{
    "attributes": [
        {
            "attribute_name": "string",
            "description": "string",
            "label": "string"
        }
    ]
}

Generate {batch_size} specific attributes covering various business contexts. Keep attributes granular and ensure each description clearly explains both purpose and contents in two distinct sentences.
"""
    return prompt.format(batch_size=batch_size)

[Rest of the script remains the same]



import json
import csv
import time
import logging
from typing import List, Dict
from datetime import datetime

# [Previous logging and rate limiting code remains the same]

def create_llm_prompt(batch_size: int) -> str:
    prompt = """Task: Generate attribute edge cases where the classification between Sensitive PII and Non-sensitive PII depends on the content detail level and included information.

[Previous context section remains the same]

Format:
Create specific attributes that clearly indicate their content. Instead of basic/detailed suffixes, name attributes based on what they contain.

Example pairs:
"investment_holdings","User's publicly traded stock names and quantity of shares held in portfolio","Non-sensitive PII"
"investment_account_credentials","User's trading account passwords, PINs, and bank account numbers for transaction authorization","Sensitive PII"
"investment_bank_accounts","User's linked bank account numbers and routing codes for trading activities","Sensitive PII"
"investment_preferences","User's preferred trading strategies and target market sectors","Non-sensitive PII"

"employee_base_salary","User's annual base compensation amount and pay grade level","Sensitive PII"
"employee_bonus","User's performance bonus amounts and payment schedules","Sensitive PII"
"employee_equity_grants","User's stock option details including grant dates and vesting schedules","Sensitive PII"
"employee_benefits_selection","User's chosen health plan types and coverage categories","Non-sensitive PII"
"employee_total_compensation","User's complete payment package including all compensation elements and tax details","Sensitive PII"

"medical_provider","User's doctor name and clinic location for appointment scheduling","Non-sensitive PII"
"medical_diagnosis","User's health condition details and prescribed treatment plans","Sensitive PII"
"medical_prescriptions","User's medication details including dosage and frequency","Sensitive PII"
"medical_appointment_time","User's scheduled visit dates and general clinic locations","Non-sensitive PII"

Each entry should follow this structure:
attribute_name,description,label

Description requirements:
- Start with "User's" or "Person's" or "Individual's"
- Provide one comprehensive sentence explaining the content and purpose
- Don't use words like "sensitive" or "non-sensitive"
- Include specific examples of included data elements
- Focus on practical, real-world usage

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

Generate {batch_size} attributes covering various business contexts, ensuring each attribute name clearly indicates its specific content.
"""
    return prompt.format(batch_size=batch_size)

[Rest of the script remains the same]








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
