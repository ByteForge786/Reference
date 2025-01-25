import pandas as pd

def split_text_column(input_file, output_file, text_column_name):
    try:
        # Read the CSV file
        df = pd.read_csv(input_file)
        
        # Function to extract attribute name and description from text
        def extract_info(text):
            try:
                # Split based on the pattern in your text column
                parts = text.split("Description:", 1)  # Adjust split pattern if needed
                
                if len(parts) == 2:
                    # Further clean attribute name
                    attr_name = parts[0].replace("Attribute Name:", "").strip()
                    description = parts[1].strip()
                else:
                    # If pattern not found, return empty values or handle as needed
                    attr_name = ""
                    description = text
                
                return pd.Series([attr_name, description])
            
            except Exception as e:
                print(f"Error processing row: {text}")
                return pd.Series(["", ""])
        
        # Create new columns by splitting the text column
        df[['attribute_name', 'enhanced_description']] = df[text_column_name].apply(extract_info)
        
        # Save to new CSV file
        df.to_csv(output_file, index=False)
        print(f"Successfully created {output_file}")
        
        # Display sample of processed data
        print("\nSample of processed data:")
        print(df.head())
        
    except Exception as e:
        print(f"Error processing CSV: {e}")

if __name__ == "__main__":
    input_file = input("Enter input CSV file path: ")
    output_file = input("Enter output CSV file path: ")
    text_column = input("Enter the name of the text column to split: ")
    split_text_column(input_file, output_file, text_column)
