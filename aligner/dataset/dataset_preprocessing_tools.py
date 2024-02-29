import pandas as pd

def generate_splits(metadata_df, train_percent, val_percent, test_percent):
    assert train_percent + val_percent + test_percent == 100, "Percentages must sum to 100."
    
    # Shuffle DataFrame rows
    shuffled_df = metadata_df.sample(frac=1).reset_index(drop=True)
    
    # Calculate split indices
    total_rows = shuffled_df.shape[0]
    train_end = int(total_rows * (train_percent / 100))
    val_end = train_end + int(total_rows * (val_percent / 100))
    
    # Assign split labels
    shuffled_df['split'] = 'test'  # Default to test
    shuffled_df.loc[:train_end, 'split'] = 'train'  # Assign rows up to train_end to train
    shuffled_df.loc[train_end:val_end, 'split'] = 'validation'  # Assign rows from train_end to val_end to validation
    
    return shuffled_df.sort_values(by='id').reset_index(drop=True)