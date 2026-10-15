import numpy as np
import pandas as pd

class MatrixFactorization:
    def __init__(self, n_factors=20, learning_rate=0.01, reg=0.1, epochs=15):
        self.n_factors = n_factors
        self.learning_rate = learning_rate
        self.reg = reg
        self.epochs = epochs
        
    def fit(self, train_ratings):
        self.train_ratings = train_ratings
        
        # 1. Map user and movie IDs to contiguous indices (0, 1, 2...)
        self.user_ids = train_ratings['userId'].unique()
        self.movie_ids = train_ratings['movieId'].unique()
        
        self.user_to_idx = {u: i for i, u in enumerate(self.user_ids)}
        self.movie_to_idx = {m: i for i, m in enumerate(self.movie_ids)}
        
        self.idx_to_movie = {i: m for m, i in self.movie_to_idx.items()}
        
        n_users = len(self.user_ids)
        n_items = len(self.movie_ids)
        
        # 2. Initialize latent vectors and biases
        # Using a fixed seed ensures your results are reproducible
        np.random.seed(42)
        self.user_factors = np.random.normal(0, 0.1, (n_users, self.n_factors))
        self.item_factors = np.random.normal(0, 0.1, (n_items, self.n_factors))
        
        self.user_biases = np.zeros(n_users)
        self.item_biases = np.zeros(n_items)
        self.global_mean = train_ratings['rating'].mean()
        
        # Convert to numpy arrays for much faster iteration during training
        users = np.array([self.user_to_idx[u] for u in train_ratings['userId']])
        items = np.array([self.movie_to_idx[m] for m in train_ratings['movieId']])
        ratings = np.array(train_ratings['rating'])
        
        # 3. Stochastic Gradient Descent (SGD)
        print("Training Matrix Factorization (this may take 10-20 seconds)...")
        for epoch in range(self.epochs):
            for u, i, r in zip(users, items, ratings):
                # Predict current rating
                prediction = self.global_mean + self.user_biases[u] + self.item_biases[i] + np.dot(self.user_factors[u], self.item_factors[i])
                
                # Calculate error
                error = r - prediction
                
                # Update biases
                self.user_biases[u] += self.learning_rate * (error - self.reg * self.user_biases[u])
                self.item_biases[i] += self.learning_rate * (error - self.reg * self.item_biases[i])
                
                # Update latent factors
                u_factor_temp = self.user_factors[u].copy()
                self.user_factors[u] += self.learning_rate * (error * self.item_factors[i] - self.reg * self.user_factors[u])
                self.item_factors[i] += self.learning_rate * (error * u_factor_temp - self.reg * self.item_factors[i])

    def recommend(self, user_id, top_k=10):
        if user_id not in self.user_to_idx:
            # Cold start user: fall back to an empty list or global mean
            return pd.DataFrame({'movieId': [], 'score': []})
            
        u_idx = self.user_to_idx[user_id]
        
        # Predict scores for all movies for this user
        # Broadcast the dot product across all item factors instantly using NumPy
        predictions = (self.global_mean + self.user_biases[u_idx] + self.item_biases + 
                       np.dot(self.item_factors, self.user_factors[u_idx]))
        
        # Create DataFrame and sort
        scores_df = pd.DataFrame({
            'movieId': self.movie_ids,
            'score': predictions
        })
        
        scores_df = scores_df.sort_values('score', ascending=False)
        return scores_df.head(top_k)

from surprise import SVD, Dataset, Reader

class SurpriseSVD:
    def __init__(self, n_factors=20, n_epochs=15, lr_all=0.01, reg_all=0.1):
        # We use the exact same hyperparameters as our custom MF for a fair fight
        self.model = SVD(n_factors=n_factors, n_epochs=n_epochs, 
                         lr_all=lr_all, reg_all=reg_all, random_state=42)
        self.train_ratings = None
        self.all_movie_ids = None

    def fit(self, train_ratings):
        self.train_ratings = train_ratings
        
        # Surprise requires a specific data format and a Reader object
        reader = Reader(rating_scale=(0.5, 5.0))
        data = Dataset.load_from_df(train_ratings[['userId', 'movieId', 'rating']], reader)
        
        # Build the training set and fit the model
        trainset = data.build_full_trainset()
        self.model.fit(trainset)
        
        # Cache unique movies to generate predictions later
        self.all_movie_ids = train_ratings['movieId'].unique()

    def recommend(self, user_id, top_k=10):
        # Predict the rating for every single movie for this user
        predictions = [
            (m_id, self.model.predict(user_id, m_id, verbose=False).est)
            for m_id in self.all_movie_ids
        ]
        
        # Sort by highest predicted rating
        predictions.sort(key=lambda x: x[1], reverse=True)
        
        # Format as a DataFrame so it works with our evaluate_model() function
        top_preds = predictions[:top_k]
        return pd.DataFrame({
            'movieId': [p[0] for p in top_preds],
            'score': [p[1] for p in top_preds]
        })