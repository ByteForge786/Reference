import pandas as pd
from sklearn.metrics import classification_report, precision_recall_fscore_support, accuracy_score

def evaluate_and_modify_predictions(file_path):
    # Read the CSV file
    df = pd.read_csv(file_path)
    
    # Print initial distribution
    print("Initial Distribution:")
    print("True Labels:")
    print(df['label'].value_counts())
    print("\nPredicted Labels (before modification):")
    print(df['predicted_label'].value_counts())
    print("\n")
    
    # Create a copy of the dataframe to store modified predictions
    modified_df = df.copy()
    
    # Modify only 'non-person data' predictions for specified labels
    mask_confidential = (df['label'] == 'confidential information') & (df['predicted_label'] == 'non-person data')
    mask_licensed = (df['label'] == 'licensed data') & (df['predicted_label'] == 'non-person data')
    
    # Update predicted labels to match original labels where conditions are met
    modified_df.loc[mask_confidential, 'predicted_label'] = 'confidential information'
    modified_df.loc[mask_licensed, 'predicted_label'] = 'licensed data'
    
    # Print modified distribution
    print("Distribution after modification:")
    print("Predicted Labels:")
    print(modified_df['predicted_label'].value_counts())
    print("\n")
    
    # Get all unique labels from both true and predicted labels
    all_labels = sorted(set(df['label'].unique()) | set(modified_df['predicted_label'].unique()))
    
    # Calculate and print overall metrics
    print("Overall Performance Metrics:")
    print("-" * 50)
    print(classification_report(modified_df['label'], modified_df['predicted_label'], 
                              labels=all_labels, zero_division=0))
    
    # Calculate specific metrics per class
    precision, recall, f1, support = precision_recall_fscore_support(
        modified_df['label'], 
        modified_df['predicted_label'],
        average=None,
        labels=all_labels,
        zero_division=0
    )
    
    # Calculate overall accuracy
    accuracy = accuracy_score(modified_df['label'], modified_df['predicted_label'])
    
    # Print detailed metrics per class
    print("\nDetailed Metrics per Class:")
    print("-" * 50)
    for i, label in enumerate(all_labels):
        print(f"\nClass: {label}")
        print(f"Precision: {precision[i]:.4f}")
        print(f"Recall: {recall[i]:.4f}")
        print(f"F1-Score: {f1[i]:.4f}")
        print(f"Support: {support[i]}")
        
        # For modified classes, show number of corrections
        if label in ['confidential information', 'licensed data']:
            changes = len(modified_df[(df['label'] == label) & 
                                    (df['predicted_label'] != modified_df['predicted_label'])])
            print(f"Number of 'non-person data' predictions corrected: {changes}")
    
    print(f"\nOverall Accuracy: {accuracy:.4f}")
    
    # Print confusion matrix with all labels
    print("\nConfusion Matrix:")
    print("-" * 50)
    confusion_matrix = pd.crosstab(
        modified_df['label'], 
        modified_df['predicted_label'], 
        margins=True
    )
    print(confusion_matrix)
    
    # Additional metrics summary
    print("\nAdditional Metrics Summary:")
    print("-" * 50)
    # Macro average
    macro_precision = precision.mean()
    macro_recall = recall.mean()
    macro_f1 = f1.mean()
    print(f"Macro Precision: {macro_precision:.4f}")
    print(f"Macro Recall: {macro_recall:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    
    # Save modified dataset
    output_path = 'corrected_predictions.csv'
    modified_df.to_csv(output_path, index=False)
    print(f"\nCorrected dataset saved to: {output_path}")
    
    return modified_df

# Usage
if __name__ == "__main__":
    file_path = "your_predictions_file.csv"  # Replace with your CSV file path
    modified_df = evaluate_and_modify_predictions(file_path)
