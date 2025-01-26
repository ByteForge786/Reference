import pandas as pd
import numpy as np
from collections import Counter

def balance_dataset(file_path):
    # Read the CSV file
    df = pd.read_csv(file_path)
    
    # Print initial statistics
    print("Initial distribution:")
    print(df['sensitivity'].value_counts())
    print("\n")
    
    # Separate the dataframe by sensitivity types
    sensitive_pii = df[df['sensitivity'] == 'sensitive_pii']
    non_sensitive_pii = df[df['sensitivity'] == 'non_sensitive_pii']
    
    # Get other classes (assuming they're not named 'sensitive_pii' or 'non_sensitive_pii')
    other_classes = df[~df['sensitivity'].isin(['sensitive_pii', 'non_sensitive_pii'])]
    
    # Calculate target size for other classes combined
    target_size = len(non_sensitive_pii)
    
    if len(other_classes) > target_size:
        # If we have more samples than needed, randomly sample
        other_classes_balanced = other_classes.sample(n=target_size, random_state=42)
    else:
        # If we have fewer samples, we'll need to oversample
        other_classes_balanced = other_classes.sample(n=target_size, replace=True, random_state=42)
    
    # Combine all parts
    balanced_df = pd.concat([sensitive_pii, non_sensitive_pii, other_classes_balanced])
    
    # Print final statistics
    print("Final distribution:")
    print(balanced_df['sensitivity'].value_counts())
    
    # Print detailed statistics
    print("\nDetailed Statistics:")
    print("-" * 50)
    print(f"Total records: {len(balanced_df)}")
    print("\nPer Label Statistics:")
    for label in balanced_df['sensitivity'].unique():
        label_data = balanced_df[balanced_df['sensitivity'] == label]
        print(f"\n{label}:")
        print(f"Count: {len(label_data)}")
        print(f"Unique attributes: {label_data['attribute_name'].nunique()}")
        print(f"Unique sources: {label_data['sourced_from'].nunique()}")
    
    # Save balanced dataset
    output_path = 'balanced_dataset.csv'
    balanced_df.to_csv(output_path, index=False)
    print(f"\nBalanced dataset saved to: {output_path}")
    
    return balanced_df

# Usage
if __name__ == "__main__":
    file_path = "your_input_file.csv"  # Replace with your CSV file path
    balanced_df = balance_dataset(file_path)
