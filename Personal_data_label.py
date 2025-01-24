import csv
from test import chinou_response

def classify_personal_data(attributes):
    """
    Process attributes marked as Personal Data and classify them as Sensitive/Non-Sensitive PII.
    Only rows with label='Personal Data' are sent for classification.
    """
    classified_data = []
    
    for attribute_name, description, label in attributes:
        if label == "Personal Data":
            prompt = f"""
Classify the following attribute based on the provided definitions:
Attribute Name: {attribute_name}
Description: {description}
Sensitive PII includes personal information that falls under the following categories:
- Racial or Ethnic origin
- Political opinions
- Religion
- Trade Union Membership
- Physical or Mental Health Data
- Genetic Data
- Biometric Data
- Sexual activities
- Criminal History (includes if victim of a crime)
- Registered Domicile Information - Japan Specific
- My Number – Japan Specific Identification Number
- Social Standing by Birth - Japan Specific
- Financial/Credit Records Information
- Unique Identification/Registration Data (passport, ID Card, Social Security Numbers, resident registration number (RRN), Licenses, alien registration number)
- Veteran Status
Non-Sensitive PII encompasses personal information that does not fall under the sensitive categories.
Please classify this attribute as either "Sensitive PII" or "Non-Sensitive PII".
"""
            try:
                classification = chinou_response(prompt)
                classified_data.append((attribute_name, description, classification))
            except Exception as e:
                print(f"Error classifying {attribute_name}: {str(e)}")
                classified_data.append((attribute_name, description, label))
        else:
            classified_data.append((attribute_name, description, label))
            
    return classified_data

def main():
    try:
        # Try different encodings if needed
        encodings = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252']
        
        for encoding in encodings:
            try:
                # Read input CSV
                with open('personal_data.csv', 'r', encoding=encoding) as file:
                    reader = csv.reader(file)
                    header = next(reader)  # Skip header row
                    attributes = [(row[0], row[1], row[2]) for row in reader]
                print(f"Successfully read file with {encoding} encoding")
                break
            except UnicodeDecodeError:
                continue
            except Exception as e:
                print(f"Error with {encoding} encoding: {str(e)}")
                continue
        else:
            raise Exception("Could not read file with any of the attempted encodings")

        # Classify the data
        classified_data = classify_personal_data(attributes)

        # Write results back to CSV with the same encoding that worked for reading
        with open('personal_data.csv', 'w', newline='', encoding=encoding) as file:
            writer = csv.writer(file)
            writer.writerow(['attribute_name', 'description', 'classification'])
            writer.writerows(classified_data)
            
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
