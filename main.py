import pandas as pd
import os

# Import modules from the src folder
from src.data import chronological_split
from src.baselines import PopularityRecommender
from src.content import ContentRecommender
from src.collaborative import MatrixFactorization, SurpriseSVD
from src.metrics import evaluate_model
from src.neural import NeuralRecommender

def main():
    print("Loading data...")
    ratings_path = "data/raw/ml-latest-small/ratings.csv" 
    
    if not os.path.exists(ratings_path):
        raise FileNotFoundError(f"Could not find {ratings_path}. Please run download.py first.")
        
    ratings = pd.read_csv(ratings_path)
    
    print("Splitting data chronologically...")
    train_data, test_data = chronological_split(ratings, test_ratio=0.2)
    print(f"Training on {len(train_data)} ratings, testing on {len(test_data)} ratings.")
    
    # 1. Popularity Baseline
    print("\n--- 1. Training Popularity Baseline ---")
    pop_model = PopularityRecommender(min_ratings_quantile=0.5)
    pop_model.fit(train_data)
    metrics = evaluate_model(pop_model, train_data, test_data, k=10, relevance_threshold=3.5)
    for metric, score in metrics.items():
        print(f"{metric}: {score:.4f}")

    # 2. Content-Based Filtering
    print("\n--- 2. Training Content-Based Model ---")
    movies_path = "data/raw/ml-latest-small/movies.csv"
    movies = pd.read_csv(movies_path)
    content_model = ContentRecommender(movies)
    content_model.fit(train_data)
    content_metrics = evaluate_model(content_model, train_data, test_data, k=10)
    for metric, score in content_metrics.items():
        print(f"{metric}: {score:.4f}")

    # 3. Custom Matrix Factorization
    print("\n--- 3. Training Custom Matrix Factorization ---")
    mf_model = MatrixFactorization(n_factors=20, epochs=15)
    mf_model.fit(train_data)
    mf_metrics = evaluate_model(mf_model, train_data, test_data, k=10)
    for metric, score in mf_metrics.items():
        print(f"{metric}: {score:.4f}")

    # 4. Surprise SVD (Industry Standard MF)
    print("\n--- 4. Training Surprise SVD Model ---")
    surprise_model = SurpriseSVD(n_factors=20, n_epochs=15)
    surprise_model.fit(train_data)
    surprise_metrics = evaluate_model(surprise_model, train_data, test_data, k=10)
    for metric, score in surprise_metrics.items():
        print(f"{metric}: {score:.4f}")

    # 5. PyTorch Neural Matrix Factorization
    print("\n--- 5. Training PyTorch Neural Model ---")
    neural_model = NeuralRecommender(n_factors=20, epochs=10)
    neural_model.fit(train_data)
    neural_metrics = evaluate_model(neural_model, train_data, test_data, k=10)
    for metric, score in neural_metrics.items():
        print(f"{metric}: {score:.4f}")

if __name__ == "__main__":
    main()