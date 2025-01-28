from setfit import SetFitModel, Trainer, TrainingArguments, sample_dataset
import pandas as pd
import numpy as np
from datasets import Dataset
import torch
from datetime import datetime
import os

def load_and_preprocess_data(file_path, min_samples=10):
    """Load and preprocess the CSV data"""
    # Read the CSV file and shuffle
    df = pd.read_csv(file_path)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Print original class distribution
    print("\nOriginal class distribution:")
    print(df['label'].value_counts())
    
    # Create training and test sets based on minimum samples per class
    train_data = []
    test_data = []
    
    for label in df['label'].unique():
        label_data = df[df['label'] == label]
        if len(label_data) >= min_samples:
            # Take minimum required samples for training
            train_data.append(label_data.head(min_samples))
            # Rest goes to testing
            test_data.append(label_data.iloc[min_samples:])
    
    train_df = pd.concat(train_data)
    test_df = pd.concat(test_data)
    
    print("\nTraining set distribution:")
    print(train_df['label'].value_counts())
    print("\nTest set distribution:")
    print(test_df['label'].value_counts())
    
    # Convert to HuggingFace datasets
    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)
    
    return train_dataset, test_dataset

def save_model(model, metrics, base_path="models"):
    """Save the trained model with metrics"""
    # Create timestamp for unique model naming
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    accuracy_str = f"{metrics['accuracy']:.4f}".replace(".", "_")
    model_path = os.path.join(base_path, f"setfit_model_acc{accuracy_str}_{timestamp}")
    
    # Create directory if it doesn't exist
    os.makedirs(base_path, exist_ok=True)
    
    # Save the model
    model.save_pretrained(model_path)
    
    # Save metadata
    metadata = {
        "metrics": metrics,
        "timestamp": timestamp
    }
    
    metadata_path = os.path.join(model_path, "metadata.txt")
    with open(metadata_path, "w") as f:
        for key, value in metadata.items():
            f.write(f"{key}: {value}\n")
    
    print(f"\nModel saved to: {model_path}")
    return model_path

def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Load and preprocess data
    train_dataset, test_dataset = load_and_preprocess_data("your_data.csv")
    
    # Initialize model
    num_classes = len(set(train_dataset['label']))
    model = SetFitModel.from_pretrained(
        "BAAI/bge-small-en-v1.5", 
        use_differentiable_head=True, 
        head_params={"out_features": num_classes}
    )
    
    # Prepare training arguments
    args = TrainingArguments(
        batch_size=(32, 16),
        num_epochs=(3, 8),
        end_to_end=True,
        body_learning_rate=(2e-5, 5e-6),
        head_learning_rate=2e-3,
        l2_weight=0.01,
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
    )
    
    # Train the model
    print("\nStarting training...")
    trainer.train()
    
    # Evaluate
    metrics = trainer.evaluate(test_dataset)
    print("\nTest metrics:", metrics)
    
    # Save the model
    save_model(model, metrics)

if __name__ == "__main__":
    main()
