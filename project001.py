import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from surprise import Dataset, Reader, KNNBasic
from surprise.model_selection import train_test_split
import tmdbsimple as tmdb
from PIL import Image
# Load data
df1 = pd.read_csv("https://raw.githubusercontent.com/hanaamulic/WBSFLIX-recommendations/main/data/ml-latest-small/movies.csv")
df2 = pd.read_csv("https://raw.githubusercontent.com/hanaamulic/WBSFLIX-recommendations/main/data/ml-latest-small/ratings.csv")
combine_df = pd.concat([df1, df2], ignore_index=True)
tmdb.API_KEY = "ac6b5e3a5f72e2d8552e4582926227a1"

def get_movie_poster_url(movie_title):
    search = tmdb.Search()
    response = search.movie(query=movie_title)
    
    # Get the first result (assuming it's the closest aamatch)
    if response['results']:
        movie_id = response['results'][0]['id']
        movie = tmdb.Movies(movie_id)
        details = movie.info()
        if 'poster_path' in details and details['poster_path']:
            # Construct the full poster URL
            poster_url = f"https://image.tmdb.org/t/p/original{details['poster_path']}"
            return poster_url
    
    # Return a placeholder URL if no poster is found
    return "https://via.placeholder.com/50"



# Popularity recommender
def popular_movie_recommender(top_n=5):
    # Calculate the rating count for each movie
    movie_rating_counts = df2['movieId'].value_counts().reset_index()
    movie_rating_counts.columns = ['movieId', 'rating_count']

    # Merge the rating count with movie information
    recommended_movies = df1.merge(movie_rating_counts, on='movieId', how='inner')
    recommended_movies = recommended_movies.sort_values(by='rating_count', ascending=False).head(top_n)
    recommended_movies['poster_url'] = recommended_movies['title'].apply(get_movie_poster_url)
    return recommended_movies[['title', 'genres', 'poster_url']]




# Function to convert rating to stars
def rating_to_stars(rating):
    if np.isnan(rating):
        return "N/A"
    else:
        return "⭐" * int(rating) + ("⭐" if rating - int(rating) >= 0.5 else "")


def main_popular_movie():
    st.title('Popular Movie Recommender')
    st.write('Welcome to the Popular Movie Recommender section!')
    top_n = st.slider('Select the number of top movies:', 1, 100, 10)

    if st.button('Get Recommendations', key="popular_rec_button"):
        recommended_movies = popular_movie_recommender(top_n)
        if not recommended_movies.empty:
            st.markdown("## Top Movie Recommendations")
            carousel = st.empty()
            for _, movie in recommended_movies.iterrows():
                if 'poster_url' in movie and movie['poster_url']:
                    # Display movie posters in a carousel
                    carousel.image(movie['poster_url'], caption=movie['title'], use_column_width=True)
                    st.markdown(f"**Title**: {movie['title']}")
                    st.markdown(f"**Genres**: {movie['genres']}")
                    st.markdown("---")  # Add a horizontal line
                else:
                    st.markdown(f"**Title**: {movie['title']} (No image available)")
                    st.markdown(f"**Genres**: {movie['genres']}")

# User-based recommender

# Collaborative filtering recommender
def collaborative_filtering_recommender(user_id, n=5):
    # Prepare data for Surprise
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(df2[['userId', 'movieId', 'rating']], reader)
    trainset, _ = train_test_split(data, test_size=0.2)

    # Build user-based collaborative filtering model
    sim_options = {
        'name': 'cosine',
        'user_based': True
    }
    model = KNNBasic(sim_options=sim_options)
    model.fit(trainset)

    # Get movie recommendations for the user
    movie_ids = [str(movie_id) for movie_id in range(1, df1['movieId'].max() + 1)]
    user_ratings = [(movie_id, model.predict(user_id, movie_id).est) for movie_id in movie_ids]
    user_ratings.sort(key=lambda x: x[1], reverse=True)
    top_n_movies = user_ratings[:n]
    
    # Get movie poster URLs
    top_n_df = pd.DataFrame(top_n_movies, columns=['movieId', 'predicted_rating'])
    top_n_df['movieId'] = top_n_df['movieId'].astype(int)
    recommended_movies = pd.merge(top_n_df, df1[['movieId', 'title']], on='movieId', how='left')
    recommended_movies['poster_url'] = recommended_movies['title'].apply(get_movie_poster_url)
    
    return recommended_movies[['title', 'predicted_rating', 'poster_url']]


# Streamlit UI for user-based movie recommender
def main_user_based_movie():
    st.title('User-Based Movie Recommender')
    st.write('Welcome to User-Based Movie Recommender Section!')
    user_id = st.number_input('Enter your user ID:', min_value=1, value=1)
    n_recommendations = st.slider('Number of recommendations:', 1, 20, 10)

    if st.button('Get Recommendations', key="user_rec_button"):
        recommended_movies = collaborative_filtering_recommender(user_id, n_recommendations)
        if recommended_movies is not None and not recommended_movies.empty:
            st.write(f'Top {n_recommendations} Recommended Movies:')
            cols = st.columns(5)  # Specify the number of columns
            for _, movie in recommended_movies.iterrows():
                with cols[0]:  # First column for images
                    st.image(movie['poster_url'], caption=movie['title'], use_column_width=True)
                with cols[1]:  # Second column for movie details
                    st.write(f"Title: {movie['title']}")
                    st.write(f"Predicted Rating: {rating_to_stars(movie['predicted_rating'])}")
        else:
            st.write('No recommendations available.')




def collaborative_filtering_recommender(user_id, n=5):
    # Filter movies with no ratings
    rated_movie_ids = df2[df2['userId'] == user_id]['movieId'].tolist()
    rated_movie_ids = set(rated_movie_ids)

    # Create a user-item matrix
    user_item_matrix = df2.pivot_table(index='userId', columns='movieId', values='rating', fill_value=0)

    # Calculate item-item similarity using cosine similarity
    item_similarity = cosine_similarity(user_item_matrix.T)

    # Get the top N similar movies that the user hasn't rated
    similar_movies = pd.Series(item_similarity[user_id], index=user_item_matrix.columns)
    similar_movies = similar_movies[~similar_movies.index.isin(rated_movie_ids)]
    similar_movies = similar_movies.sort_values(ascending=False).index[:n]

    # Get movie details and poster URLs
    recommended_movies = df1[df1['movieId'].isin(similar_movies)]
    recommended_movies['predicted_rating'] = np.nan
    recommended_movies['poster_url'] = recommended_movies['title'].apply(get_movie_poster_url)

    return recommended_movies[['title', 'predicted_rating', 'poster_url']]



# Streamlit UI for collaborative filtering
def main_collaborative_filtering():
    st.title('User ID Movie Recommender')
    st.write('Welcome to User ID Movie Recommender section!')

    user_id = st.number_input('Enter your user ID:', min_value=1, value=1)
    n_recommendations = st.slider('Number of recommendations:', 1, 20, 10)

    if st.button('Get Recommendations', key="collab_rec_button"):
        recommended_movies = collaborative_filtering_recommender(user_id, n=n_recommendations)
        if not recommended_movies.empty:
            st.markdown("## User ID")
            carousel = st.empty()
            for _, movie in recommended_movies.iterrows():
                if 'poster_url' in movie and movie['poster_url']:
                    # Display movie posters in a carousel
                    carousel.image(movie['poster_url'], caption=movie['title'], use_column_width=True)
                    st.markdown(f"**Title**: {movie['title']}")
                    st.markdown("---")  # Add a horizontal line
                else:
                    st.markdown(f"**Title**: {movie['title']} (No image available)")


def main():
    st.sidebar.title('Movie Recommender')
    app_mode = st.sidebar.selectbox("Choose the Recommender", ["Popular Movies", "User-Based", "User ID"])

    if app_mode == "Popular Movies":
        main_popular_movie()
    elif app_mode == "User-Based":
        main_user_based_movie()
    elif app_mode == "User ID":
        main_collaborative_filtering()

if __name__ == "__main__":
    main()