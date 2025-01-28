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
    
    # Filter classes with minimum samples
    label_counts = df['label'].value_counts()
    valid_labels = label_counts[label_counts >= MIN_SAMPLES_PER_CLASS].index
    df_filtered = df[df['label'].isin(valid_labels)]
    
    # Print filtered class distribution
    print("\nFiltered class distribution (min samples per class):")
    print(df_filtered['label'].value_counts())
    
    return df_filtered

def prepare_datasets(df):
    """Prepare train and test datasets"""
    # Split the data
    train_df, test_df = train_test_split(
        df, 
        test_size=TEST_SIZE, 
        stratify=df['label'],
        random_state=RANDOM_STATE
    )
    
    # Convert to HuggingFace datasets
    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)
    
    return train_dataset, test_dataset

def train_setfit_model(train_dataset, test_dataset):
    """Train the SetFit model"""
    # Initialize model
    model = SetFitModel.from_pretrained(MODEL_NAME)
    
    # Initialize trainer
    trainer = SetFitTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        column_mapping={"text": "text", "label": "label"},
        batch_size=16,
        num_iterations=20,  # Number of text pairs to generate for contrastive learning
        num_epochs=3  # Number of epochs to train
    )
    
    # Train the model
    trainer.train()
    
    # Evaluate
    metrics = trainer.evaluate()
    print("\nEvaluation metrics:")
    print(metrics)
    
    return trainer.model

def save_model(model, base_path="models"):
    """Save the trained model"""
    # Create timestamp for unique model naming
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(base_path, f"setfit_model_{timestamp}")
    
    # Create directory if it doesn't exist
    os.makedirs(base_path, exist_ok=True)
    
    # Save the model
    model.save_pretrained(model_path)
    print(f"\nModel saved to: {model_path}")
    
    return model_path

def main():
    # Set random seed for reproducibility
    torch.manual_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    
    # Load and preprocess data
    df = load_and_preprocess_data("your_data.csv")
    
    # Prepare datasets
    train_dataset, test_dataset = prepare_datasets(df)
    
    # Train model
    model = train_setfit_model(train_dataset, test_dataset)
    
    # Save model
    model_path = save_model(model)

if __name__ == "__main__":
    main()
