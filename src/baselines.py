import pandas as pd

class PopularityRecommender:
    def __init__(self, min_ratings_quantile=0.5):
        """
        min_ratings_quantile: The percentile of rating counts to use as a strict cutoff.
        If set to 0.5, a movie must have more ratings than 50% of the movies in the dataset.
        """
        self.min_ratings_quantile = min_ratings_quantile
        self.global_mean = 0
        self.min_ratings = 0
        self.movie_scores = None

    def fit(self, train_ratings):
        # 1. Calculate stats per movie
        movie_stats = train_ratings.groupby('movieId').agg(
            rating_count=('rating', 'count'),
            rating_mean=('rating', 'mean')
        ).reset_index()

        # 2. Calculate global metrics for the Bayesian formula
        self.global_mean = train_ratings['rating'].mean()
        self.min_ratings = movie_stats['rating_count'].quantile(self.min_ratings_quantile)

        # 3. Filter out movies that don't meet the minimum rating threshold
        qualified_movies = movie_stats[movie_stats['rating_count'] >= self.min_ratings].copy()

        # 4. Apply the Bayesian Average Formula
        # (v / (v + m) * R) + (m / (v + m) * C)
        # v = rating count, m = min ratings, R = movie mean, C = global mean
        
        v = qualified_movies['rating_count']
        m = self.min_ratings
        R = qualified_movies['rating_mean']
        C = self.global_mean

        qualified_movies['bayesian_score'] = (v / (v + m) * R) + (m / (v + m) * C)

        # 5. Sort by highest score to establish the global ranking
        self.movie_scores = qualified_movies.sort_values('bayesian_score', ascending=False)

    def recommend(self, user_id, top_k=10):
        """
        Returns the top K movies. Since this is a non-personalized baseline, 
        it returns the same list for every user.
        """
        if self.movie_scores is None:
            raise ValueError("Model has not been trained. Call fit() first.")
        
        return self.movie_scores.head(top_k)[['movieId', 'bayesian_score']]