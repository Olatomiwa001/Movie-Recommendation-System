import streamlit as st
import sqlite3
from movie_recommender import (
    initialize_database,
    get_user_ratings,
    content_based_recommendations,
    collaborative_filtering_recommendations,
    hybrid_recommendations,
    add_movie,
    add_user,
)

# Initialize the database
conn = initialize_database()

# Streamlit App Configuration
st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Sidebar Branding
st.sidebar.image(
    "https://upload.wikimedia.org/wikipedia/commons/6/69/IMDB_Logo_2016.svg",
    width=200,
    caption="Powered by IMDb",
)

# App Title
st.title("🎬 Movie Recommendation System")
st.markdown(
    """
    Welcome to the **Movie Recommendation System**!  
    Discover movies you'll love using **Content-Based**, **Collaborative Filtering**, and **Hybrid** recommendation techniques.
    """
)

# Sidebar Navigation
menu = st.sidebar.radio(
    "Navigation",
    [
        "Home",
        "User Ratings",
        "Content-Based Recommendations",
        "Collaborative Filtering Recommendations",
        "Hybrid Recommendations",
        "Add a Movie",
        "Add a User",
        "About the Project",
    ],
)

# Home Page
if menu == "Home":
    st.header("Welcome to the Movie Recommendation System!")
    st.markdown(
        """
        ### Features:
        - **User Ratings**: View ratings given by users.
        - **Content-Based Recommendations**: Get recommendations based on movie content.
        - **Collaborative Filtering Recommendations**: Discover movies based on similar users.
        - **Hybrid Recommendations**: Combine both methods for better results.
        - **Add Movies and Users**: Expand the database with new entries.
        """
    )

# User Ratings
elif menu == "User Ratings":
    st.header("⭐ User Ratings")
    user_id = st.number_input("Enter User ID:", min_value=1, step=1)
    if st.button("Show Ratings"):
        ratings = get_user_ratings(conn, user_id)
        if ratings is not None and not ratings.empty:
            st.dataframe(ratings)
        else:
            st.warning(f"No ratings found for User ID {user_id}.")

# Content-Based Recommendations
elif menu == "Content-Based Recommendations":
    st.header("🎭 Content-Based Recommendations")
    movie_title = st.text_input("Enter Movie Title:")
    if st.button("Get Recommendations"):
        recommendations = content_based_recommendations(conn, movie_title)
        if recommendations:
            st.write("Recommended Movies:")
            st.dataframe(recommendations)
        else:
            st.warning(f"No recommendations found for '{movie_title}'.")

# Collaborative Filtering Recommendations
elif menu == "Collaborative Filtering Recommendations":
    st.header("🤝 Collaborative Filtering Recommendations")
    user_id = st.number_input("Enter User ID:", min_value=1, step=1)
    if st.button("Get Recommendations"):
        recommendations = collaborative_filtering_recommendations(conn, user_id)
        if recommendations:
            st.write("Recommended Movies:")
            st.dataframe(recommendations)
        else:
            st.warning(f"No recommendations found for User ID {user_id}.")

# Hybrid Recommendations
elif menu == "Hybrid Recommendations":
    st.header("🔀 Hybrid Recommendations")
    user_id = st.number_input("Enter User ID:", min_value=1, step=1)
    movie_title = st.text_input("Enter Movie Title (optional):")
    if st.button("Get Recommendations"):
        recommendations = hybrid_recommendations(conn, user_id, movie_title if movie_title else None)
        if recommendations:
            st.write("Recommended Movies:")
            st.dataframe(recommendations)
        else:
            st.warning(f"No recommendations found for User ID {user_id}.")

# Add a Movie
elif menu == "Add a Movie":
    st.header("🎥 Add a New Movie")
    title = st.text_input("Movie Title:")
    genre = st.text_input("Genre:")
    year = st.number_input("Year:", min_value=1900, max_value=2100, step=1, value=2025)
    poster_url = st.text_input("Poster URL (optional):")
    popularity = st.number_input("Popularity:", min_value=0.0, max_value=10.0, step=0.1, value=0.0)
    if st.button("Add Movie"):
        add_movie(conn, title, genre, year, poster_url, popularity)
        st.success(f"Movie '{title}' added successfully!")

# Add a User
elif menu == "Add a User":
    st.header("👤 Add a New User")
    username = st.text_input("Username:")
    if st.button("Add User"):
        user_id = add_user(conn, username)
        st.success(f"User '{username}' added successfully!")

# About the Project
elif menu == "About the Project":
    st.header("📚 About the Project")
    st.markdown(
        """
        **Movie Recommendation System** is designed to help users discover movies they might enjoy based on different recommendation approaches:
        - **Content-Based Filtering**: Suggests movies similar to those a user has liked before.
        - **Collaborative Filtering**: Recommends movies based on what similar users have watched and rated.
        - **Hybrid Approach**: Combines both methods for more accurate recommendations.
        
        ### Why This Project?
        - To explore the **power of SQL and Python** in data management and analysis.
        - To implement **machine learning techniques** in recommendation systems.
        - To provide an **interactive and user-friendly** way to explore movies.
        
        This project is built with **Streamlit** for UI, **SQLite** for the database, and **Python** for the recommendation logic.
        """
    )
