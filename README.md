#### Movie_Rec
Movie_Recommenders
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from surprise import Dataset, Reader, KNNBasic
from surprise.model_selection import train_test_split
import tmdbsimple as tmdb
from PIL import Image
#### Load data
df1 = pd.read_csv("https://raw.githubusercontent.com/hanaamulic/WBSFLIX-recommendations/main/data/ml-latest-small/movies.csv")
df2 = pd.read_csv("https://raw.githubusercontent.com/hanaamulic/WBSFLIX-recommendations/main/data/ml-latest-small/ratings.csv")
combine_df = pd.concat([df1, df2], ignore_index=True)
tmdb.API_KEY = "YOUR_API"

def get_movie_poster_url(movie_title):
    search = tmdb.Search()
    response = search.movie(query=movie_title)
    
    # Get the first result (assuming it's the closest match)
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


#### Popularity recommender
def popular_movie_recommender(top_n=5):
    # Implementation of the popularity-based movie recommender
    combine_df['rating_count'] = combine_df.groupby('movieId')['rating'].transform('count')
    combine_df['average_rating'] = combine_df.groupby('movieId')['rating'].transform('mean')
    
    # Filter out movies with missing title, genres, or rating
    recommended_movies = combine_df.dropna(subset=['title', 'genres', 'rating_count', 'average_rating'])
    
    recommended_movies = recommended_movies.sort_values(by=['rating_count', 'average_rating'], ascending=False).head(top_n)
    
    # Get movie poster URLs
    recommended_movies['poster_url'] = recommended_movies['title'].apply(get_movie_poster_url)
    
    return recommended_movies[['title', 'genres', 'rating_count', 'average_rating', 'poster_url']]

####streamlit ui
def main_popular_movie():
    st.title('Popular Movie Recommender')
    st.write('Welcome to the Popular Movie Recommender section!')
    top_n = st.slider('Select the number of top movies:', 1, 100, 10)

    if st.button('Get Recommendations', key="popular_rec_button"):
        recommended_movies = popular_movie_recommender(top_n)

        if not recommended_movies.empty:
            st.write(f'Top {top_n} Recommended Movies:')
            for _, movie in recommended_movies.iterrows():
                st.image(movie['poster_url'], caption=movie['title'], use_column_width=True)
                st.write(f"Genres: {movie['genres']}, Rating Count: {movie['rating_count']}, Average Rating: {movie['average_rating']}")
        else:
            st.write('No recommendations available.')



#### User-based recommender
def user_based_movie_recommender(user_id, n_recommendations):
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
    top_n_movies = user_ratings[:n_recommendations]
    
    # Get movie poster URLs
    top_n_df = pd.DataFrame(top_n_movies, columns=['movieId', 'predicted_rating'])
    top_n_df['movieId'] = top_n_df['movieId'].astype(int)
    recommended_movies = pd.merge(top_n_df, df1[['movieId', 'title']], on='movieId', how='left')
    recommended_movies['poster_url'] = recommended_movies['title'].apply(get_movie_poster_url)
    
    return recommended_movies[['title', 'predicted_rating', 'poster_url']]


#### Streamlit UI for user-based movie recommender
def main_user_based_movie():
    st.title('User-Based Movie Recommender')
    st.write('Welcome to User-Based Movie Recommender Section!')

    user_id = st.number_input('Enter your user ID:', min_value=1, value=1)
    n_recommendations = st.slider('Number of recommendations:', 1, 20, 10)

    if st.button('Get Recommendations', key="user_rec_button"):
        recommended_movies = user_based_movie_recommender(user_id, n_recommendations)

        if not recommended_movies.empty:
            st.write(f'Top {n_recommendations} Recommended Movies:')
            for _, movie in recommended_movies.iterrows():
                st.image(movie['poster_url'], caption=movie['title'], use_column_width=True)
                st.write(f"Predicted Rating: {movie['predicted_rating']}")
        else:
            st.write('No recommendations available.')

#### Streamlit user_id
    # Placeholder implementation for collaborative filtering
    # Replace this with the actual collaborative filtering algorithm
    # ...# Streamlit user_id

def collaborative_filtering_recommender(user_id, n=5):
    # Placeholder implementation for collaborative filtering
    # Replace this with the actual collaborative filtering algorithm
    user_item_matrix = combine_df.pivot_table(index='userId', columns='movieId', values='rating', fill_value=0)
    item_similarity = cosine_similarity(user_item_matrix.T)
    item_similarity_df = pd.DataFrame(item_similarity, index=user_item_matrix.columns, columns=user_item_matrix.columns)
    
    # Find top N similar movies
    similar_movies = item_similarity_df[user_id].sort_values(ascending=False).index[1:n+1]
    
    # Get recommended movies
    recommended_movies_df = df1[df1['movieId'].isin(similar_movies)]
    return recommended_movies_df

#### Streamlit UI for collaborative filtering
def main_collaborative_filtering():
    st.title('User ID Movie Recommender')
    st.write('Welcome to User ID Movie Recommender section!')

    user_id = st.number_input('Enter your user ID:', min_value=1, value=1)

    if st.button('Get Recommendations', key="User_ID"):
        recommended_movies_df = collaborative_filtering_recommender(user_id, n=5)


        if not recommended_movies_df.empty:
            st.write('Recommended Movies:')
            st.dataframe(recommended_movies_df)
        else:
            st.write('No recommendations available.')

#### Streamlit main application
def main():
    st.sidebar.title('Movie Recommender')
    app_mode = st.sidebar.selectbox("Choose the Recommender", ["User ID", "Popular Movies", "User-Based"])

    if app_mode == "User ID":
        main_collaborative_filtering()
    elif app_mode == "Popular Movies":
        main_popular_movie()
    elif app_mode == "User-Based":
        main_user_based_movie()

if __name__ == "__main__":
    main()
