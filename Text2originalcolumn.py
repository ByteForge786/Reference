import pandas as pd

def process_csv_fast(input_file, output_file, text_column_name):
    try:
        # Read CSV
        df = pd.read_csv(input_file)
        
        # Extract attribute name
        df['attribute_name'] = (df[text_column_name]
                              .str.split('Attribute Name:', expand=True)[1]
                              .str.split('Description:', expand=True)[0]
                              .str.strip())
        
        # Extract description
        df['enhanced_description'] = (df[text_column_name]
                                    .str.split('Description:', expand=True)[1]
                                    .str.split('Consider the privacy impact', expand=True)[0]
                                    .str.strip())
        
        # Save result
        df.to_csv(output_file, index=False)
        print(f"Successfully processed {len(df)} rows")
        
        # Quick sample check
        print("\nFirst row sample:")
        print(f"Attribute Name: {df['attribute_name'].iloc[0]}")
        print(f"Description: {df['enhanced_description'].iloc[0]}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    input_file = input("Input CSV path: ")
    output_file = input("Output CSV path: ")
    text_column = input("Text column name: ")
    process_csv_fast(input_file, output_file, text_column)
