import pandas as pd
from transformers import T5TokenizerFast

def create_classification_prompt(row):
    """Create a detailed prompt combining attribute name and description."""
    return (
        f"Given the following data attribute details, classify it into one of these categories:\n"
        f"- Sensitive PII (personal data that could lead to fraud/harm if public)\n"
        f"- Non-sensitive PII (personal data that's generally safe to be public)\n"
        f"- Non-person data (internal company data, not related to individuals)\n\n"
        f"Attribute Name: {row['attribute_name']}\n"
        f"Description: {row['enhanced_description']}\n"
        f"Classify this attribute as:"
    )

def process_csv(input_file, output_file, max_tokens=512):
    try:
        # Load tokenizer
        tokenizer = T5TokenizerFast.from_pretrained('google/flan-t5-base')
        
        # Read the input CSV
        df = pd.read_csv(input_file)
        
        # Create the combined text prompt
        df['text'] = df.apply(create_classification_prompt, axis=1)
        
        # Check token lengths
        def check_tokens(text):
            tokens = tokenizer.encode(text)
            return len(tokens)
        
        # Create new dataframe with text and label columns
        output_df = pd.DataFrame({
            'text': df['text'],
            'label': df['classification']
        })
        
        # Check token lengths before saving
        token_lengths = [check_tokens(text) for text in output_df['text']]
        if max(token_lengths) > max_tokens:
            print(f"Warning: Some prompts exceed {max_tokens} tokens. Longest prompt: {max(token_lengths)} tokens")
        
        # Save to new CSV file
        output_df.to_csv(output_file, index=False)
        print(f"\nSuccessfully created {output_file}")
        
        # Print token statistics
        print(f"\nToken statistics:")
        print(f"Max tokens in any prompt: {max(token_lengths)}")
        print(f"Average tokens per prompt: {sum(token_lengths)/len(token_lengths):.1f}")
        print(f"Number of prompts: {len(token_lengths)}")
        
        # Show a sample prompt
        print(f"\nSample prompt:")
        print(output_df['text'].iloc[0])
        
    except Exception as e:
        print(f"Error processing CSV: {e}")

if __name__ == "__main__":
    input_file = input("Enter input CSV file path: ")
    output_file = input("Enter output CSV file path: ")
    process_csv(input_file, output_file)
