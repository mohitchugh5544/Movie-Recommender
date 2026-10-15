import pandas as pd
import itertools
import os

from src.data import chronological_split
from src.neural import NeuralRecommender
from src.metrics import evaluate_model

def run_grid_search():
    print("Loading data...")
    ratings = pd.read_csv("data/raw/ml-latest-small/ratings.csv")
    
    # 1. Standard Split (Hold out the final 20% for the absolute final test)
    train_full, test_data = chronological_split(ratings, test_ratio=0.2)
    
    # 2. Validation Split (Split the training data AGAIN to create a tuning set)
    print("Creating validation set for tuning...")
    train_tune, val_tune = chronological_split(train_full, test_ratio=0.2)
    
    # 3. Define the Hyperparameter Grid
    param_grid = {
        'n_factors': [16, 32, 64],
        'lr': [0.001, 0.005, 0.01],
        'weight_decay': [0.01, 0.05]
    }
    
    # Extract keys and values to create combinations
    keys = param_grid.keys()
    values = param_grid.values()
    combinations = list(itertools.product(*values))
    
    best_score = 0
    best_params = None
    results = []
    
    print(f"\nStarting Grid Search: Testing {len(combinations)} combinations...\n")
    
    # 4. The Tuning Loop
    for combo in combinations:
        # Zip the keys with the current combination's values into a dictionary
        params = dict(zip(keys, combo))
        print(f"Testing parameters: {params}")
        
        # Initialize model with the current parameters
        # We use fewer epochs (e.g., 5) to keep the search fast
        model = NeuralRecommender(
            n_factors=params['n_factors'], 
            lr=params['lr'], 
            weight_decay=params['weight_decay'], 
            epochs=5
        )
        
        # Train on the tuning-training set
        model.fit(train_tune)
        
        # Evaluate on the tuning-validation set
        metrics = evaluate_model(model, train_tune, val_tune, k=10)
        ndcg_score = metrics['NDCG@10']
        
        print(f"Result -> NDCG@10: {ndcg_score:.4f} | Precision@10: {metrics['Precision@10']:.4f}\n")
        
        # Store results
        params['NDCG@10'] = ndcg_score
        results.append(params)
        
        # Track the best performing model
        if ndcg_score > best_score:
            best_score = ndcg_score
            best_params = params

    # 5. Output Summary
    print("-" * 30)
    print(f"Best NDCG@10: {best_score:.4f}")
    print(f"Best Parameters: {best_params}")
    
    # Optional: Save all results to a CSV to analyze trends later
    results_df = pd.DataFrame(results)
    results_df.to_csv("data/processed/tuning_results.csv", index=False)
    print("Full results saved to data/processed/tuning_results.csv")

if __name__ == "__main__":
    run_grid_search()