from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score
import pickle
import os

def load_model_and_head(model_path):
    """Load both the transformer model and classifier head"""
    print(f"Loading model from: {model_path}")
    try:
        # Load sentence transformer
        model = SentenceTransformer(model_path)
        
        # Load the classifier head
        head_path = os.path.join(model_path, "model_head.pkl")
        with open(head_path, 'rb') as f:
            head = pickle.load(f)
            
        print("Model and classifier head loaded successfully")
        return model, head
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

def load_label_mapping(model_path):
    """Load label mapping from metadata"""
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

def predict_texts(model, head, texts, id_to_label):
    """Make predictions for texts"""
    # Get embeddings
    embeddings = model.encode(texts)
    
    # Get predictions and probabilities
    probabilities = head.predict_proba(embeddings)
    predictions = head.predict(embeddings)
    
    # Convert numeric predictions to labels
    predicted_labels = [id_to_label[pred] for pred in predictions]
    confidences = np.max(probabilities, axis=1)
    
    return predicted_labels, confidences

def predict_single(model, head, text, id_to_label):
    """Handle single text prediction"""
    predicted_labels, confidences = predict_texts(model, head, [text], id_to_label)
    
    print("\nPrediction Results:")
    print(f"Text: {text}")
    print(f"Predicted Label: {predicted_labels[0]}")
    print(f"Confidence: {confidences[0]:.4f}")
    
    return {
        'predicted_label': predicted_labels[0],
        'confidence': confidences[0]
    }

def predict_csv(model, head, file_path, id_to_label, label_mapping):
    """Handle CSV batch prediction"""
    # Read CSV
    df = pd.read_csv(file_path)
    
    if 'text' not in df.columns:
        raise ValueError("CSV must contain a 'text' column")
    
    texts = df['text'].tolist()
    has_labels = 'label' in df.columns
    
    # Get predictions
    print("\nGenerating predictions...")
    predicted_labels, confidences = predict_texts(model, head, texts, id_to_label)
    
    # Add predictions to dataframe
    df['predicted_label'] = predicted_labels
    df['confidence'] = confidences
    
    # Calculate metrics if true labels exist
    if has_labels:
        print("\nClassification Report:")
        print(classification_report(df['label'], df['predicted_label']))
        
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
    
    # Save results
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
    
    # Load model, head and label mapping
    model, head = load_model_and_head(args.model_path)
    id_to_label, label_mapping = load_label_mapping(args.model_path)
    
    # Process input
    if args.input.endswith('.csv'):
        results = predict_csv(model, head, args.input, id_to_label, label_mapping)
    else:
        results = predict_single(model, head, args.input, id_to_label)

if __name__ == "__main__":
    main()
