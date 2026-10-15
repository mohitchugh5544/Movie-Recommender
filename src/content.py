import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

class ContentRecommender:
    def __init__(self, movies_df):
        self.movies_df = movies_df.copy()
        self.sim_matrix = None
        self.movie_idx_map = None
        self.train_ratings = None

    def fit(self, train_ratings):
        self.train_ratings = train_ratings
        
        # 1. Clean genres for TF-IDF (e.g., "Action|Adventure" -> "Action Adventure")
        self.movies_df['genres_clean'] = self.movies_df['genres'].str.replace('|', ' ')
        
        # 2. Vectorize the genres
        tfidf = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf.fit_transform(self.movies_df['genres_clean'])
        
        # 3. Compute cosine similarity matrix for all movies
        # linear_kernel is equivalent to cosine_similarity when using TF-IDF, but much faster
        self.sim_matrix = linear_kernel(tfidf_matrix, tfidf_matrix)
        
        # 4. Map movieId to the matrix index for fast lookups
        self.movie_idx_map = pd.Series(
            self.movies_df.index, index=self.movies_df['movieId']
        ).to_dict()

    def recommend(self, user_id, top_k=10):
        # 1. Get movies the user liked in their training history
        user_history = self.train_ratings[
            (self.train_ratings['userId'] == user_id) & 
            (self.train_ratings['rating'] >= 3.5)
        ]
        
        # Handle "Cold Start" users with no positive training history
        if user_history.empty:
            return pd.DataFrame({'movieId': [], 'score': []})
            
        liked_movie_ids = user_history['movieId'].tolist()
        
        # Get matrix indices for the movies they liked
        liked_indices = [
            self.movie_idx_map[m_id] for m_id in liked_movie_ids 
            if m_id in self.movie_idx_map
        ]
        
        if not liked_indices:
            return pd.DataFrame({'movieId': [], 'score': []})

        # 2. Calculate average similarity of all movies against the ones they liked
        # This creates a personalized "taste profile" vector
        taste_profile_scores = np.mean(self.sim_matrix[liked_indices], axis=0)
        
        # 3. Format and sort the top scores
        scores_df = pd.DataFrame({
            'movieId': self.movies_df['movieId'],
            'score': taste_profile_scores
        })
        
        scores_df = scores_df.sort_values('score', ascending=False)
        
        return scores_df.head(top_k)