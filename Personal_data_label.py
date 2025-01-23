import csv
from test import chinou_response

def classify_personal_data(attributes):
    classified_attributes = []
    for name, description, label in attributes:
        if label == "Personal Data":
            prompt = f"""
Classify the following attribute based on the provided definitions:

Attribute Name: {name}
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
            classification = chinou_response(prompt)
            classified_attributes.append((name, classification))
        else:
            classified_attributes.append((name, label))
    return classified_attributes

def main():
    with open('personal_data.csv', 'r') as file:
        reader = csv.reader(file)
        header = next(reader)
        rows = [row for row in reader]

    classified_rows = []
    for row in rows:
        attributes = list(zip(header, row, ['Personal Data' if i in [13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30] else 'Non-Sensitive PII' for i in range(len(row))]))
        classified_attributes = classify_personal_data(attributes)
        classified_row = [classification if name in [col[0] for col in classified_attributes if col[1] == 'Sensitive PII'] else value for name, value, classification in classified_attributes]
        classified_rows.append(classified_row)

    with open('personal_data.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(classified_rows)

if __name__ == "__main__":
    main()
