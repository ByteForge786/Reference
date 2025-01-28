from setfit import SetFitModel
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, accuracy_score
import torch
import torch.nn.functional as F
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

def get_confidence_scores(model, texts):
    """Get prediction probabilities using softmax"""
    # Get model outputs before softmax
    with torch.no_grad():
        outputs = model.model_body(texts)
        logits = model.model_head(outputs)
        # Apply softmax to get probabilities
        probs = F.softmax(logits, dim=1)
    return probs.numpy()

def predict_single_text(model, text, id_to_label):
    """Handle single text prediction"""
    # Get prediction and confidence
    probs = get_confidence_scores(model, [text])
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

def predict_csv(model, file_path, id_to_label, label_mapping):
    """Handle CSV batch prediction and evaluation"""
    # Read CSV
    df = pd.read_csv(file_path)
    
    if 'text' not in df.columns:
        raise ValueError("CSV must contain a 'text' column")
    
    texts = df['text'].tolist()
    has_labels = 'label' in df.columns
    
    # Get predictions and confidence scores
    probs = get_confidence_scores(model, texts)
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
    
    # Load model and label mapping
    model = SetFitModel.from_pretrained(args.model_path)
    id_to_label, label_mapping = load_label_mapping(args.model_path)
    
    # Determine input type and process accordingly
    if args.input.endswith('.csv'):
        # Batch prediction on CSV
        results = predict_csv(model, args.input, id_to_label, label_mapping)
    else:
        # Single text prediction
        results = predict_single_text(model, args.input, id_to_label)

if __name__ == "__main__":
    main()
