import pandas as pd

def chronological_split(df, test_ratio=0.2):
    """
    Sorts the data by timestamp and reserves the most recent ratings 
    of each user for the test set.
    """
    # Sort by timestamp to ensure chronological order
    df_sorted = df.sort_values(by=['userId', 'timestamp'])
    
    # Calculate the split index for each user
    def split_user(group):
        cutoff = int(len(group) * (1 - test_ratio))
        return group.iloc[:cutoff], group.iloc[cutoff:]
    
    # Apply split and recombine
    splits = df_sorted.groupby('userId', group_keys=False).apply(split_user)
    
    # Unpack the tuple of DataFrames
    train_df = pd.concat([res[0] for res in splits])
    test_df = pd.concat([res[1] for res in splits])
    
    return train_df, test_df