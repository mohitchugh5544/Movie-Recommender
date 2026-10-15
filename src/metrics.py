import numpy as np
import pandas as pd

def precision_at_k(recommended_items, relevant_items, k=10):
    """
    Measures the proportion of the top-K recommendations that are actually relevant.
    """
    recommended_at_k = recommended_items[:k]
    # Count how many recommended items are in the user's relevant set
    hits = len(set(recommended_at_k).intersection(set(relevant_items)))
    
    return hits / k if k > 0 else 0.0

def ndcg_at_k(recommended_items, relevant_items, k=10):
    """
    Normalized Discounted Cumulative Gain at K.
    Rewards models heavily for putting relevant items at the very top of the ranking.
    """
    recommended_at_k = recommended_items[:k]
    
    dcg = 0.0
    for i, item in enumerate(recommended_at_k):
        if item in relevant_items:
            # i is 0-indexed. Position is i+1. Formula uses log2(position + 1) -> log2(i + 2)
            dcg += 1.0 / np.log2(i + 2)
            
    # Calculate Ideal DCG (if all relevant items were ranked perfectly at the top)
    idcg = 0.0
    num_ideal_hits = min(len(relevant_items), k)
    for i in range(num_ideal_hits):
        idcg += 1.0 / np.log2(i + 2)
        
    return dcg / idcg if idcg > 0.0 else 0.0

def evaluate_model(model, train_df, test_df, k=10, relevance_threshold=3.5):
    """
    Generates recommendations for all users in the test set and averages their scores.
    """
    # Filter test set for items the user actually liked
    relevant_test_data = test_df[test_df['rating'] >= relevance_threshold]
    user_relevant_items = relevant_test_data.groupby('userId')['movieId'].apply(set).to_dict()
    
    # We only evaluate users who actually have relevant items in the test set
    eval_users = list(user_relevant_items.keys())
    
    precisions = []
    ndcgs = []
    
    for user_id in eval_users:
        # Get true relevant items for this user
        relevant_items = user_relevant_items[user_id]
        
        # Get model recommendations (ignoring movies the user saw in the training set)
        user_train_history = set(train_df[train_df['userId'] == user_id]['movieId'])
        
        # Get raw recommendations
        raw_recs = model.recommend(user_id, top_k=k + len(user_train_history))
        recs_list = raw_recs['movieId'].tolist()
        
        # Filter out items the user already rated in the training set
        novel_recs = [item for item in recs_list if item not in user_train_history][:k]
        
        # Calculate metrics
        precisions.append(precision_at_k(novel_recs, relevant_items, k))
        ndcgs.append(ndcg_at_k(novel_recs, relevant_items, k))
        
    return {
        f'Precision@{k}': np.mean(precisions),
        f'NDCG@{k}': np.mean(ndcgs)
    }