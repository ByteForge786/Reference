import pandas as pd
from transformers import T5TokenizerFast

def create_classification_prompt(row):
    """Create a prompt with reasoning for classification."""
    return (
        f"Classify this data attribute into one of these categories:\n"
        f"- Sensitive PII: user data that if made public can harm user through fraud or theft\n"
        f"- Non-sensitive PII: user data that can be safely made public without harm\n"
        f"- Non-person data: internal company data not related to personal information\n\n"
        f"Attribute Name: {row['attribute_name']}\n"
        f"Description: {row['enhanced_description']}\n"
        f"Consider the privacy impact and potential for misuse. Classify this as:"
    )

def process_csv(input_file, output_file, max_tokens=512):
    try:
        # Load tokenizer
        tokenizer = T5TokenizerFast.from_pretrained('google/flan-t5-small')
        
        # Read the input CSV
        df = pd.read_csv(input_file)
        
        # Create the combined text prompt
        df['text'] = df.apply(create_classification_prompt, axis=1)
        
        # Create new dataframe with text and label columns
        output_df = pd.DataFrame({
            'text': df['text'],
            'label': df['classification']
        })
        
        # Check token lengths
        token_lengths = [len(tokenizer.encode(text)) for text in output_df['text']]
        
        # Save to new CSV file
        output_df.to_csv(output_file, index=False)
        print(f"Successfully created {output_file}")
        
        # Print sample and stats
        print("\nSample prompt:")
        print(output_df['text'].iloc[0])
        print(f"\nToken statistics:")
        print(f"Max tokens in any prompt: {max(token_lengths)}")
        print(f"Average tokens per prompt: {sum(token_lengths)/len(token_lengths):.1f}")
        
    except Exception as e:
        print(f"Error processing CSV: {e}")

if __name__ == "__main__":
    input_file = input("Enter input CSV file path: ")
    output_file = input("Enter output CSV file path: ")
    process_csv(input_file, output_file)
