import streamlit as st
import pandas as pd
from src.data import chronological_split
from src.neural import NeuralRecommender

# @st.cache_resource ensures the model only trains ONCE when the app starts,
# keeping the UI instantly responsive afterward.
@st.cache_resource
def load_and_train_model():
    # Load data
    movies = pd.read_csv("data/raw/ml-latest-small/movies.csv")
    ratings = pd.read_csv("data/raw/ml-latest-small/ratings.csv")
    
    # We only need the training data for the live app
    train_data, _ = chronological_split(ratings, test_ratio=0.2)
    
    # Train the PyTorch model
    model = NeuralRecommender(n_factors=20, epochs=10)
    model.fit(train_data)
    
    return movies, train_data, model

# --- UI Layout ---
st.title("🎬 AI Movie Recommender")
st.write("Powered by PyTorch Neural Matrix Factorization")

# Load everything behind a loading spinner
with st.spinner("Waking up the AI (training takes ~15 seconds)..."):
    movies_df, train_df, model = load_and_train_model()

# Create a sidebar or main area for user input
user_ids = sorted(train_df['userId'].unique())
selected_user = st.selectbox("Select a User ID to generate recommendations:", user_ids)

if st.button("Generate Recommendations"):
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Past Favorites")
        st.write("Movies this user rated 4.0 or higher:")
        
        # Find their top rated movies
        user_history = train_df[(train_df['userId'] == selected_user) & (train_df['rating'] >= 4.0)]
        liked_movies = pd.merge(user_history, movies_df, on='movieId')
        
        # Display as a clean table
        st.dataframe(
            liked_movies[['title', 'genres', 'rating']].head(10), 
            hide_index=True,
            use_container_width=True
        )

    with col2:
        st.subheader("🤖 AI Recommends")
        st.write("Top 10 predicted movies:")
        
        # Get PyTorch Predictions
        recs = model.recommend(selected_user, top_k=10)
        
        # Merge with movie metadata to get titles
        rec_details = pd.merge(recs, movies_df, on='movieId')
        
        # Display as a clean table
        st.dataframe(
            rec_details[['title', 'genres', 'score']], 
            hide_index=True,
            use_container_width=True
        )