import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Movie Recommender", layout="wide")
st.title("🎬 Movie Recommender System")

@st.cache_data
def load_data():
    ratings = pd.read_csv("data/u.data", sep="\t", header=None,
                          names=['user_id', 'movie_id', 'rating', 'timestamp'])

    movies = pd.read_csv("data/u.item", sep='|', encoding='latin-1', header=None,
                         names=['movie_id', 'movie_title', 'release_date', 'video_release_date', 'IMDb_URL',
                                'unknown', 'Action', 'Adventure', 'Animation', "Children's", 'Comedy',
                                'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical',
                                'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'],
                         usecols=list(range(24)))

    data = pd.merge(ratings, movies, on='movie_id')
    user_movie_matrix = data.pivot_table(index='user_id', columns='movie_title', values='rating')

    similarity = cosine_similarity(user_movie_matrix.fillna(0))
    similarity_df = pd.DataFrame(similarity, index=user_movie_matrix.index, columns=user_movie_matrix.index)

    try:
        imdb_df = pd.read_csv("data/imdb_ratings.csv")
        movies = movies.merge(imdb_df, on="IMDb_URL", how="left")
    except FileNotFoundError:
        movies["imdb_rating"] = np.nan

    user_names = {i: f'User {i}' for i in user_movie_matrix.index}
    return data, user_movie_matrix, similarity_df, movies, user_names

data, user_movie_matrix, user_similarity_df, movies_df, user_names = load_data()

def predict_rating(user_id, movie_title):
    if movie_title not in user_movie_matrix.columns:
        return None
    sim_scores = user_similarity_df[user_id]
    movie_ratings = user_movie_matrix[movie_title]
    valid = movie_ratings.notna()
    sim_scores = sim_scores[valid]
    movie_ratings = movie_ratings[valid]
    if len(sim_scores) == 0:
        return None
    prediction = np.dot(sim_scores, movie_ratings) / sim_scores.sum()
    return round(prediction, 2)

def recommend_movies(user_id, n=5):
    user_rated = user_movie_matrix.loc[user_id]
    unrated_movies = user_rated[user_rated.isna()].index  # Only movies not rated by the user
    predictions = [(movie, predict_rating(user_id, movie)) for movie in unrated_movies]
    predictions = [p for p in predictions if p[1] is not None]
    predictions.sort(key=lambda x: x[1], reverse=True)
    return predictions[:n]

st.header("🎯 Collaborative Filtering")
user_names_list = list(user_names.values())
selected_user_name = st.selectbox("Select a User", user_names_list)
selected_user_id = list(user_names.keys())[list(user_names.values()).index(selected_user_name)]

if st.button("Recommend Movies", key="collab_recommend_btn"):
    with st.spinner("Generating recommendations..."):
        recommendations = recommend_movies(selected_user_id)
        if recommendations:
            st.success("Top recommended movies:")
            for i, (movie, score) in enumerate(recommendations, 1):
                avg_rating = data[data['movie_title'] == movie]['rating'].mean()
                imdb_row = movies_df[movies_df['movie_title'] == movie]
                imdb_url = imdb_row['IMDb_URL'].values[0] if not imdb_row.empty else None
                imdb_rating = imdb_row['imdb_rating'].values[0] if not imdb_row.empty else 'N/A'

                st.write(
                    f"{i}. {movie} — "
                    f"⭐ Predicted: {score} | "
                    f"Avg. User Rating: {round(avg_rating,2) if not np.isnan(avg_rating) else 'N/A'} | "
                    f"IMDb Rating: {imdb_rating if not pd.isnull(imdb_rating) else 'N/A'}"
                    + (f" | [IMDb Link]({imdb_url})" if imdb_url else "")
                )
        else:
            st.warning("No recommendations available.")

# Test button to show seen movies for the selected user
if st.button("Show Seen Movies", key="seen_movies_btn"):
    seen_movies = user_movie_matrix.loc[selected_user_id].dropna().index.tolist()
    st.info("Seen movies by this user:")
    for i, m in enumerate(seen_movies, 1):
        st.write(f"{i}. {m}")

st.divider()

# Genre-based Recommender
st.header("📂 Genre-Based Recommender")
genre_columns = ['Action', 'Adventure', 'Animation', "Children's", 'Comedy', 'Crime', 'Documentary',
                 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery',
                 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
selected_genres = st.multiselect("Select genre(s) you like:", genre_columns)
recommend_genre_clicked = st.button("Recommend by Genre", key="genre_recommend_btn")
if recommend_genre_clicked:
    if selected_genres:
        with st.spinner("Recommending..."):
            genre_filter = movies_df[selected_genres].sum(axis=1) == len(selected_genres)
            filtered_movies = movies_df[genre_filter]
            movie_ratings = data.groupby('movie_title')['rating'].mean().reset_index()
            movie_ratings.columns = ['movie_title', 'avg_rating']
            merged = pd.merge(filtered_movies, movie_ratings, on='movie_title')
            top_movies = merged.sort_values(by='avg_rating', ascending=False).head(5)
            st.success("Top movies in selected genre(s):")
            for i, row in top_movies.iterrows():
                imdb_row = movies_df[movies_df['movie_title'] == row['movie_title']]
                imdb_url = imdb_row['IMDb_URL'].values[0] if not imdb_row.empty else None
                imdb_rating = imdb_row['imdb_rating'].values[0] if not imdb_row.empty else 'N/A'
                st.write(
                    f"{i+1}. {row['movie_title']} — "
                    f"⭐ Avg. Rating: {round(row['avg_rating'],2)} | "
                    f"IMDb Rating: {imdb_rating if not pd.isnull(imdb_rating) else 'N/A'}"
                    + (f" | [IMDb Link]({imdb_url})" if imdb_url else "")
                )
    else:
        st.warning("Please select at least one genre.")
