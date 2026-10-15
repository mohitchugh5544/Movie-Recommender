import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class MFNet(nn.Module):
    """The PyTorch neural network architecture for Matrix Factorization."""
    def __init__(self, num_users, num_items, n_factors=20):
        super(MFNet, self).__init__()
        # User and Item latent feature vectors
        self.user_embedding = nn.Embedding(num_users, n_factors)
        self.item_embedding = nn.Embedding(num_items, n_factors)
        
        # User and Item biases
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_bias = nn.Embedding(num_items, 1)
        
        # Initialize weights with a small variance
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)

    def forward(self, user, item):
        # Dot product of user and item embeddings
        dot_product = (self.user_embedding(user) * self.item_embedding(item)).sum(dim=1, keepdim=True)
        
        # Add biases
        prediction = dot_product + self.user_bias(user) + self.item_bias(item)
        return prediction.squeeze()

class NCFNet(nn.Module):
    """Neural Collaborative Filtering: Concatenates embeddings and uses an MLP."""
    def __init__(self, num_users, num_items, n_factors=32, hidden_dim=64):
        super(NCFNet, self).__init__()
        # 1. Embeddings (slightly larger to feed the network)
        self.user_embedding = nn.Embedding(num_users, n_factors)
        self.item_embedding = nn.Embedding(num_items, n_factors)
        
        # 2. Multi-Layer Perceptron (MLP)
        # Input size is n_factors * 2 because we concatenate user and item vectors
        self.fc1 = nn.Linear(n_factors * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.output = nn.Linear(hidden_dim // 2, 1)
        
        # 3. Activation & Regularization
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, user, item):
        # Grab the vectors
        u_emb = self.user_embedding(user)
        i_emb = self.item_embedding(item)
        
        # Concatenate them side-by-side
        x = torch.cat([u_emb, i_emb], dim=1)
        
        # Pass through the deep network
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        # Output final score
        prediction = self.output(x)
        return prediction.squeeze()
    
class NeuralRecommender:
    """Wrapper class to match the fit/recommend API of our other models."""
    def __init__(self, n_factors=20, epochs=10, batch_size=512, lr=0.005, weight_decay=0.01):
        self.n_factors = n_factors
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def fit(self, train_ratings):
        # 1. Map IDs to 0-indexed integers
        self.user_ids = train_ratings['userId'].unique()
        self.movie_ids = train_ratings['movieId'].unique()
        
        self.user_to_idx = {u: i for i, u in enumerate(self.user_ids)}
        self.movie_to_idx = {m: i for i, m in enumerate(self.movie_ids)}
        self.idx_to_movie = {i: m for m, i in self.movie_to_idx.items()}
        
        # 2. Prepare PyTorch Tensors and DataLoader
        users = torch.tensor([self.user_to_idx[u] for u in train_ratings['userId']], dtype=torch.long)
        items = torch.tensor([self.movie_to_idx[m] for m in train_ratings['movieId']], dtype=torch.long)
        ratings = torch.tensor(train_ratings['rating'].values, dtype=torch.float32)
        
        # Center ratings around the global mean to help the network learn faster
        self.global_mean = ratings.mean().item()
        ratings = ratings - self.global_mean
        
        dataset = TensorDataset(users, items, ratings)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # 3. Initialize Model, Loss, and Optimizer
        self.model = MFNet(len(self.user_ids), len(self.movie_ids), self.n_factors).to(self.device)
        criterion = nn.MSELoss()
        # AdamW adds L2 regularization (weight decay) to prevent overfitting
        optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        
        # 4. Training Loop
        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            for batch_users, batch_items, batch_ratings in dataloader:
                batch_users, batch_items, batch_ratings = batch_users.to(self.device), batch_items.to(self.device), batch_ratings.to(self.device)
                
                optimizer.zero_grad()
                predictions = self.model(batch_users, batch_items)
                loss = criterion(predictions, batch_ratings)
                
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

    def recommend(self, user_id, top_k=10):
        if user_id not in self.user_to_idx:
            return pd.DataFrame({'movieId': [], 'score': []})
            
        self.model.eval()
        u_idx = self.user_to_idx[user_id]
        
        # Create tensors for this user and ALL possible items
        user_tensor = torch.tensor([u_idx] * len(self.movie_ids), dtype=torch.long).to(self.device)
        item_tensor = torch.tensor(list(self.movie_to_idx.values()), dtype=torch.long).to(self.device)
        
        with torch.no_grad():
            # Add the global mean back to the predictions
            predictions = self.model(user_tensor, item_tensor).cpu().numpy() + self.global_mean
            
        scores_df = pd.DataFrame({
            'movieId': self.movie_ids,
            'score': predictions
        })
        
        scores_df = scores_df.sort_values('score', ascending=False)
        return scores_df.head(top_k)