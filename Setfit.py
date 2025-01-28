import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from setfit import SetFitModel, SetFitTrainer
from datasets import Dataset
import torch
from datetime import datetime
import os

# Configuration
MIN_SAMPLES_PER_CLASS = 10  # Minimum number of samples per class
TEST_SIZE = 0.2
RANDOM_STATE = 42
MODEL_NAME = "sentence-transformers/paraphrase-mpnet-base-v2"  # You can change this to other models

def load_and_preprocess_data(file_path):
    """Load and preprocess the CSV data"""
    # Read the CSV file
    df = pd.read_csv(file_path)
    
    # Print original class distribution
    print("\nOriginal class distribution:")
    print(df['label'].value_counts())
    
    # Create training and test sets based on minimum samples per class
    train_data = []
    test_data = []
    
    for label in df['label'].unique():
        label_data = df[df['label'] == label]
        if len(label_data) >= MIN_SAMPLES_PER_CLASS:
            # Take minimum required samples for training
            train_data.append(label_data.head(MIN_SAMPLES_PER_CLASS))
            # Rest goes to testing
            test_data.append(label_data.iloc[MIN_SAMPLES_PER_CLASS:])
    
    train_df = pd.concat(train_data)
    test_df = pd.concat(test_data)
    
    print("\nTraining set distribution (minimum samples per class):")
    print(train_df['label'].value_counts())
    print("\nTest set distribution (remaining samples):")
    print(test_df['label'].value_counts())
    
    return train_df, test_df

def prepare_datasets(train_df, test_df):
    """Prepare train and test datasets"""
    # Print dataset sizes
    print("\nDataset sizes:")
    print(f"Training set size: {len(train_df)} ({MIN_SAMPLES_PER_CLASS} samples per class)")
    print(f"Test set size: {len(test_df)} (remaining samples)")
    
    # Convert to HuggingFace datasets
    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)
    
    return train_dataset, test_dataset

def train_setfit_model(train_dataset, test_dataset):
    """Train the SetFit model with best model saving"""
    # Initialize model
    model = SetFitModel.from_pretrained(MODEL_NAME)
    
    # Initialize trainer with validation monitoring
    trainer = SetFitTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        column_mapping={"text": "text", "label": "label"},
        batch_size=16,
        num_iterations=20,  # Number of text pairs to generate for contrastive learning
        num_epochs=3,  # Number of epochs to train
        metric="accuracy",  # Metric to track for best model
    )
    
    # Variables to track best model
    best_accuracy = 0.0
    best_model = None
    
    # Custom training loop with validation
    for epoch in range(trainer.num_epochs):
        print(f"\nEpoch {epoch + 1}/{trainer.num_epochs}")
        
        # Train for one epoch
        trainer.train_epoch()
        
        # Evaluate
        metrics = trainer.evaluate()
        current_accuracy = metrics["accuracy"]
        print(f"Validation Accuracy: {current_accuracy:.4f}")
        
        # Save best model
        if current_accuracy > best_accuracy:
            best_accuracy = current_accuracy
            best_model = SetFitModel.from_pretrained(MODEL_NAME)
            best_model.model_body.load_state_dict(trainer.model.model_body.state_dict())
            best_model.model_head.load_state_dict(trainer.model.model_head.state_dict())
            print(f"New best model saved! Accuracy: {best_accuracy:.4f}")
    
    print(f"\nTraining completed. Best accuracy: {best_accuracy:.4f}")
    return best_model

def save_model(model, accuracy, base_path="models"):
    """Save the best model with its accuracy score"""
    # Create timestamp for unique model naming
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    accuracy_str = f"{accuracy:.4f}".replace(".", "_")
    model_path = os.path.join(base_path, f"setfit_model_acc{accuracy_str}_{timestamp}")
    
    # Create directory if it doesn't exist
    os.makedirs(base_path, exist_ok=True)
    
    # Save the model
    model.save_pretrained(model_path)
    
    # Save metadata
    metadata = {
        "accuracy": accuracy,
        "timestamp": timestamp,
        "model_name": MODEL_NAME,
        "min_samples_per_class": MIN_SAMPLES_PER_CLASS
    }
    
    metadata_path = os.path.join(model_path, "metadata.txt")
    with open(metadata_path, "w") as f:
        for key, value in metadata.items():
            f.write(f"{key}: {value}\n")
    
    print(f"\nBest model saved to: {model_path}")
    print(f"Accuracy: {accuracy:.4f}")
    
    return model_path

def main():
    # Set random seed for reproducibility
    torch.manual_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    
    # Load and preprocess data
    train_df, test_df = load_and_preprocess_data("your_data.csv")
    
    # Prepare datasets
    train_dataset, test_dataset = prepare_datasets(train_df, test_df)
    
    # Train model
    model = train_setfit_model(train_dataset, test_dataset)
    
    # Save model
    model_path = save_model(model)

if __name__ == "__main__":
    main()
