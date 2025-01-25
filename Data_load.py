import os
import pandas as pd
from datasets import Dataset

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
label2id = {
    "sensitive pii": 0, 
    "non sensitive pii": 1, 
    "non-person data": 2
}
id2label = {id: label for label, id in label2id.items()}

def load_dataset(model_type: str = "") -> Dataset:
    """Load dataset with shuffling first, then capping at 300 per label for training."""
    
    # Read the dataset
    dataset_pandas = pd.read_csv(
        ROOT_DIR + "/data/classification_data.csv",
        names=["label", "text"],
    )
    
    # Convert labels and shuffle FIRST
    dataset_pandas["label"] = dataset_pandas["label"].astype(str)
    dataset_pandas["text"] = dataset_pandas["text"].astype(str)
    
    # Shuffle the entire dataset
    dataset_pandas = dataset_pandas.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Convert labels to ids if needed
    if model_type == "AutoModelForSequenceClassification":
        dataset_pandas["label"] = dataset_pandas["label"].map(label2id)
    
    # Now split into train and test after shuffling
    train_data = []
    test_data = []
    
    # Track count per label using the original string labels
    label_counts = {label: 0 for label in label2id.keys()}
    
    # Go through shuffled data once and assign to train/test
    for _, row in dataset_pandas.iterrows():
        current_label = row['label']
        # Convert numeric label back to string for counting if needed
        if model_type == "AutoModelForSequenceClassification":
            counting_label = id2label[current_label]
        else:
            counting_label = current_label
            
        if label_counts[counting_label] < 300:  # First 300 go to train
            train_data.append(row)
            label_counts[counting_label] += 1
        else:  # Rest go to test
            test_data.append(row)
    
    # Convert lists to dataframes
    train_dataset = pd.DataFrame(train_data)
    test_dataset = pd.DataFrame(test_data)
    
    # Convert to HuggingFace Dataset format
    dataset_dict = {
        'train': Dataset.from_pandas(train_dataset),
        'test': Dataset.from_pandas(test_dataset)
    }
    
    # Print split statistics
    print("\nDataset Split Statistics:")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    print("\nTraining set label distribution:")
    print(train_dataset["label"].value_counts())
    print("\nTest set label distribution:")
    print(test_dataset["label"].value_counts())
    
    return dataset_dict

if __name__ == "__main__":
    dataset = load_dataset()
    print("\nFull dataset structure:")
    print(dataset)
