import sqlite3
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate
import requests
import json
import os
import datetime
import csv
from io import StringIO
import sys
sys.path.append('c:/Users/User/Desktop/Movie Recommendation System')
from movie_recommender import (
    initialize_database,
    get_user_ratings,
    content_based_recommendations,
    collaborative_filtering_recommendations,
    hybrid_recommendations,
    add_movie,
    add_user,
)


# Database Management
def backup_database():
    """Create a backup of the database"""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs('backups', exist_ok=True)
    
    try:
        conn = sqlite3.connect('movie_recommendations.db')
        backup_conn = sqlite3.connect(f'backups/movie_recommendations_{timestamp}.db')
        conn.backup(backup_conn)
        conn.close()
        backup_conn.close()
        print(f"Database backed up successfully to backups/movie_recommendations_{timestamp}.db")
        return True
    except Exception as e:
        print(f"Backup failed: {e}")
        return False

def export_to_csv():
    """Export database tables to CSV files"""
    conn = sqlite3.connect('movie_recommendations.db')
    
    # Create exports directory if it doesn't exist
    os.makedirs('exports', exist_ok=True)
    
    # Export movies
    movies_df = pd.read_sql_query("SELECT * FROM movies", conn)
    movies_df.to_csv('exports/movies.csv', index=False)
    
    # Export ratings
    ratings_df = pd.read_sql_query("SELECT * FROM ratings", conn)
    ratings_df.to_csv('exports/ratings.csv', index=False)
    
    # Export users
    users_df = pd.read_sql_query("SELECT * FROM users", conn)
    users_df.to_csv('exports/users.csv', index=False)
    
    conn.close()
    print("Data exported to CSV files in 'exports' folder")
    return True

def import_from_csv():
    """Import data from CSV files to database"""
    if not os.path.exists('exports/movies.csv') or not os.path.exists('exports/ratings.csv'):
        print("CSV files not found in 'exports' folder")
        return False
    
    conn = sqlite3.connect('movie_recommendations.db')
    cursor = conn.cursor()
    
    # Clear existing data
    cursor.execute("DELETE FROM ratings")
    cursor.execute("DELETE FROM movies")
    cursor.execute("DELETE FROM users")
    
    # Import movies
    movies_df = pd.read_csv('exports/movies.csv')
    movies_df.to_sql('movies', conn, if_exists='append', index=False)
    
    # Import users
    users_df = pd.read_csv('exports/users.csv')
    users_df.to_sql('users', conn, if_exists='append', index=False)
    
    # Import ratings
    ratings_df = pd.read_csv('exports/ratings.csv')
    ratings_df.to_sql('ratings', conn, if_exists='append', index=False)
    
    conn.commit()
    conn.close()
    print("Data imported from CSV files successfully")
    return True

# Data Access Functions
def get_all_movies(conn):
    """Get all movies from database"""
    query = "SELECT * FROM movies ORDER BY popularity DESC"
    return pd.read_sql_query(query, conn)

def get_all_ratings(conn):
    """Get all ratings from database"""
    query = "SELECT * FROM ratings"
    return pd.read_sql_query(query, conn)

def search_movies(conn, term):
    """Search for movies by title or genre"""
    query = f"""
    SELECT * FROM movies 
    WHERE title LIKE '%{term}%' OR genre LIKE '%{term}%'
    ORDER BY popularity DESC
    """
    return pd.read_sql_query(query, conn)

# Data Management Functions
def add_user_rating(conn, user_id, movie_id, rating):
    """Add or update a user's movie rating"""
    cursor = conn.cursor()
    
    # Check if rating already exists
    cursor.execute('''
    SELECT rating_id FROM ratings
    WHERE user_id = ? AND movie_id = ?
    ''', (user_id, movie_id))
    
    existing = cursor.fetchone()
    
    if existing:
        # Update existing rating
        cursor.execute('''
        UPDATE ratings
        SET rating = ?, timestamp = CURRENT_TIMESTAMP
        WHERE user_id = ? AND movie_id = ?
        ''', (rating, user_id, movie_id))
        print(f"Updated rating for movie ID {movie_id}")
    else:
        # Insert new rating
        cursor.execute('''
        INSERT INTO ratings (user_id, movie_id, rating)
        VALUES (?, ?, ?)
        ''', (user_id, movie_id, rating))
        print(f"Added new rating for movie ID {movie_id}")
    
    conn.commit()
    return True

def delete_movie(conn, movie_id):
    """Delete a movie and its ratings from the database"""
    cursor = conn.cursor()
    
    # First delete associated ratings
    cursor.execute('''
    DELETE FROM ratings
    WHERE movie_id = ?
    ''', (movie_id,))
    
    # Then delete the movie
    cursor.execute('''
    DELETE FROM movies
    WHERE movie_id = ?
    ''', (movie_id,))
    
    conn.commit()
    print(f"Deleted movie ID {movie_id} and its ratings")
    return True

def delete_rating(conn, rating_id):
    """Delete a specific rating"""
    cursor = conn.cursor()
    cursor.execute('''
    DELETE FROM ratings
    WHERE rating_id = ?
    ''', (rating_id,))
    
    conn.commit()
    print(f"Deleted rating ID {rating_id}")
    return True

def update_movie(conn, movie_id, title=None, genre=None, year=None, poster_url=None, popularity=None):
    """Update movie information"""
    cursor = conn.cursor()
    
    # Get current movie data
    cursor.execute('''
    SELECT title, genre, year, poster_url, popularity
    FROM movies
    WHERE movie_id = ?
    ''', (movie_id,))
    
    current = cursor.fetchone()
    if not current:
        print(f"Movie ID {movie_id} not found")
        return False
    
    # Use current values if not provided
    title = title if title is not None else current[0]
    genre = genre if genre is not None else current[1]
    year = year if year is not None else current[2]
    poster_url = poster_url if poster_url is not None else current[3]
    popularity = popularity if popularity is not None else current[4]
    
    cursor.execute('''
    UPDATE movies
    SET title = ?, genre = ?, year = ?, poster_url = ?, popularity = ?
    WHERE movie_id = ?
    ''', (title, genre, year, poster_url, popularity, movie_id))
    
    conn.commit()
    print(f"Updated movie ID {movie_id}")
    return True

def bulk_import_movies(conn, csv_text):
    """Import multiple movies from CSV text"""
    try:
        # Parse CSV data
        df = pd.read_csv(StringIO(csv_text))
        required_columns = ['title', 'genre']
        
        # Check required columns
        if not all(col in df.columns for col in required_columns):
            print("CSV must contain 'title' and 'genre' columns")
            return False
        
        # Add optional columns if missing
        if 'year' not in df.columns:
            df['year'] = None
        if 'popularity' not in df.columns:
            df['popularity'] = 0
        if 'poster_url' not in df.columns:
            df['poster_url'] = ''
        
        # Insert movies
        cursor = conn.cursor()
        for _, row in df.iterrows():
            cursor.execute('''
            INSERT INTO movies (title, genre, year, poster_url, popularity)
            VALUES (?, ?, ?, ?, ?)
            ''', (row['title'], row['genre'], row['year'], row['poster_url'], row['popularity']))
        
        conn.commit()
        print(f"Successfully imported {len(df)} movies from CSV")
        return True
    except Exception as e:
        print(f"Error importing movies: {e}")
        return False

# Data Visualization Functions
def visualize_ratings_heatmap(conn, user_id=None):
    """Visualize user-movie ratings as a heatmap"""
    ratings_df = get_all_ratings(conn)
    movies_df = get_all_movies(conn)
    
    # Filter by user if specified
    if user_id:
        ratings_df = ratings_df[ratings_df['user_id'] == user_id]
    
    # Create pivot table of ratings
    pivot_df = ratings_df.pivot_table(index='user_id', columns='movie_id', values='rating')
    
    # Create a figure
    plt.figure(figsize=(12, 8))
    
    # Create heatmap
    ax = sns.heatmap(pivot_df, cmap='YlGnBu', annot=True, fmt='.1f', 
                     linewidths=0.5, vmin=0, vmax=5, cbar_kws={'label': 'Rating'})
    
    # Replace movie IDs with titles (for first 10 movies to avoid crowding)
    movie_labels = []
    for movie_id in pivot_df.columns[:10]:
        title = movies_df[movies_df['movie_id'] == movie_id]['title'].values[0]
        movie_labels.append(f"{movie_id}: {title[:15]}...")
    
    # Set x-axis labels for the first 10 movies only
    ax.set_xticklabels(movie_labels, rotation=45, ha='right')
    
    plt.title('User-Movie Ratings Heatmap')
    plt.tight_layout()
    
    # Save the figure
    os.makedirs('visualizations', exist_ok=True)
    plt.savefig('visualizations/ratings_heatmap.png')
    
    plt.close()
    print("Heatmap saved to visualizations/ratings_heatmap.png")
    return True

def visualize_genre_distribution(conn):
    """Visualize distribution of movie genres in database"""
    movies_df = get_all_movies(conn)
    
    # Extract all genres
    all_genres = []
    for genres in movies_df['genre'].str.split(', '):
        all_genres.extend(genres)
    
    # Count genre occurrences
    genre_counts = pd.Series(all_genres).value_counts()
    
    # Create a figure
    plt.figure(figsize=(10, 6))
    
    # Create bar chart
    sns.barplot(x=genre_counts.index, y=genre_counts.values, palette='viridis')
    
    plt.title('Distribution of Movie Genres')
    plt.xlabel('Genre')
    plt.ylabel('Count')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Save the figure
    os.makedirs('visualizations', exist_ok=True)
    plt.savefig('visualizations/genre_distribution.png')
    
    plt.close()
    print("Genre distribution chart saved to visualizations/genre_distribution.png")
    return True

def visualize_recommendations(user_id, recommendations):
    """Visualize recommendation scores as a bar chart"""
    if not recommendations:
        print("No recommendations to visualize")
        return False
    
    # Create dataframe from recommendations
    rec_df = pd.DataFrame(recommendations)
    
    # Create a figure
    plt.figure(figsize=(10, 6))
    
    # Create bar chart
    bars = plt.bar(rec_df['title'], rec_df['score'], color='skyblue')
    
    # Add source labels if available
    if 'source' in rec_df.columns:
        for i, bar in enumerate(bars):
            source = rec_df.iloc[i]['source']
            color = 'blue' if source == 'Collaborative' else 'green'
            bar.set_color(color)
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='blue', label='Collaborative Filtering'),
            Patch(facecolor='green', label='Content-based')
        ]
        plt.legend(handles=legend_elements)
    
    plt.title(f'Movie Recommendations for User {user_id}')
    plt.xlabel('Movie')
    plt.ylabel('Score')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Save the figure
    os.makedirs('visualizations', exist_ok=True)
    plt.savefig(f'visualizations/recommendations_user_{user_id}.png')
    
    plt.close()
    print(f"Recommendations chart saved to visualizations/recommendations_user_{user_id}.png")
    return True

# User Interface Functions
def clear_screen():
    """Clear the terminal screen"""
    os.system('cls' if os.name == 'nt' else 'clear')

def display_table(data, headers):
    """Display data in a formatted table"""
    if isinstance(data, pd.DataFrame):
        data_list = data.values.tolist()
    else:
        data_list = data
    
    print(tabulate(data_list, headers=headers, tablefmt="grid"))

def show_movies(conn, limit=20):
    """Show all movies in a nice table format"""
    movies_df = get_all_movies(conn)
    if len(movies_df) > limit:
        movies_df = movies_df.head(limit)
        print(f"Showing top {limit} movies (ordered by popularity)")
    
    # Display as table
    display_table(
        movies_df[['movie_id', 'title', 'genre', 'year', 'popularity']],
        ['ID', 'Title', 'Genre', 'Year', 'Popularity']
    )

def show_user_ratings(conn, user_id):
    """Show ratings for a specific user"""
    ratings = get_user_ratings(conn, user_id)
    
    if len(ratings) == 0:
        print(f"User {user_id} has not rated any movies yet")
        return
    
    print(f"\nRatings for User {user_id}:")
    display_table(
        ratings[['rating_id', 'movie_id', 'title', 'rating', 'genre']],
        ['Rating ID', 'Movie ID', 'Title', 'Rating', 'Genre']
    )

def show_recommendations(recommendations, rec_type):
    """Display recommendations in a formatted table"""
    if not recommendations:
        print("No recommendations available")
        return
    
    print(f"\n{rec_type} Recommendations:")
    
    if rec_type == "Content-based":
        display_table(
            [[r['movie_id'], r['title'], r['genre'], r.get('year', 'Unknown'), f"{r['similarity_score']:.2f}"] 
             for r in recommendations],
            ['Movie ID', 'Title', 'Genre', 'Year', 'Similarity Score']
        )
    elif rec_type == "Collaborative":
        display_table(
            [[r['movie_id'], r['title'], r['genre'], r.get('year', 'Unknown'), f"{r['predicted_rating']:.2f}"] 
             for r in recommendations],
            ['Movie ID', 'Title', 'Genre', 'Year', 'Predicted Rating']
        )
    else:  # Hybrid
        display_table(
            [[r['movie_id'], r['title'], r['genre'], r['year'], f"{r['score']:.2f}", r['source']] 
             for r in recommendations],
            ['Movie ID', 'Title', 'Genre', 'Year', 'Score', 'Source']
        )

def import_from_api():
    """Import popular movies from an external source"""
    try:
        # This is a simple simulation - in a real app, you'd use a real API like TMDB
        print("Simulating API import of popular movies...")
        
        # Sample data that would come from an API
        sample_data = """
        [
            {"id": 101, "title": "Avengers: Endgame", "genre": "Action, Adventure, Sci-Fi", "year": 2019, "popularity": 8.4},
            {"id": 102, "title": "Joker", "genre": "Crime, Drama, Thriller", "year": 2019, "popularity": 8.4},
            {"id": 103, "title": "Parasite", "genre": "Comedy, Drama, Thriller", "year": 2019, "popularity": 8.6},
            {"id": 104, "title": "The Irishman", "genre": "Biography, Crime, Drama", "year": 2019, "popularity": 7.9},
            {"id": 105, "title": "Once Upon a Time in Hollywood", "genre": "Comedy, Drama", "year": 2019, "popularity": 7.6}
        ]
        """
        
        movie_data = json.loads(sample_data)
        
        conn = sqlite3.connect('movie_recommendations.db')
        cursor = conn.cursor()
        
        # Import each movie
        for movie in movie_data:
            cursor.execute('''
            INSERT OR IGNORE INTO movies (movie_id, title, genre, year, popularity)
            VALUES (?, ?, ?, ?, ?)
            ''', (movie['id'], movie['title'], movie['genre'], movie['year'], movie['popularity']))
        
        conn.commit()
        conn.close()
        
        print(f"Imported {len(movie_data)} movies from API")
        return True
    
    except Exception as e:
        print(f"Error importing from API: {e}")
        return False

def main():
    """Main function to run the movie recommendation system"""
    conn = initialize_database()
    
    while True:
        clear_screen()
        print("Movie Recommendation System")
        print("1. Show all movies")
        print("2. Show user ratings")
        print("3. Get content-based recommendations")
        print("4. Get collaborative filtering recommendations")
        print("5. Get hybrid recommendations")
        print("6. Add a new movie")
        print("7. Add a new user")
        print("8. Add or update a user rating")
        print("9. Export database to CSV")
        print("10. Import database from CSV")
        print("11. Backup database")
        print("12. Visualize ratings heatmap")
        print("13. Visualize genre distribution")
        print("14. Import movies from API")
        print("15. Bulk import movies from CSV")
        print("16. Exit")
        
        choice = input("Enter your choice: ")
        
        if choice == "1":
            show_movies(conn)
        elif choice == "2":
            user_id = int(input("Enter user ID: "))
            show_user_ratings(conn, user_id)
        elif choice == "3":
            movie_title = input("Enter movie title: ")
            recommendations = content_based_recommendations(conn, movie_title)
            show_recommendations(recommendations, "Content-based")
        elif choice == "4":
            user_id = int(input("Enter user ID: "))
            recommendations = collaborative_filtering_recommendations(conn, user_id)
            show_recommendations(recommendations, "Collaborative")
        elif choice == "5":
            user_id = int(input("Enter user ID: "))
            movie_title = input("Enter movie title (optional, press Enter to skip): ")
            recommendations = hybrid_recommendations(conn, user_id, movie_title if movie_title else None)
            show_recommendations(recommendations, "Hybrid")
        elif choice == "6":
            title = input("Enter movie title: ")
            genre = input("Enter movie genre: ")
            year = input("Enter movie year (optional): ")
            poster_url = input("Enter poster URL (optional): ")
            popularity = float(input("Enter movie popularity (optional, default 0): ") or 0)
            add_movie(conn, title, genre, year, poster_url, popularity)
        elif choice == "7":
            username = input("Enter username: ")
            add_user(conn, username)
        elif choice == "8":
            user_id = int(input("Enter user ID: "))
            movie_id = int(input("Enter movie ID: "))
            rating = float(input("Enter rating (0-5): "))
            add_user_rating(conn, user_id, movie_id, rating)
        elif choice == "9":
            export_to_csv()
        elif choice == "10":
            import_from_csv()
        elif choice == "11":
            backup_database()
        elif choice == "12":
            user_id = input("Enter user ID (optional, press Enter to skip): ")
            visualize_ratings_heatmap(conn, int(user_id) if user_id else None)
        elif choice == "13":
            visualize_genre_distribution(conn)
        elif choice == "14":
            import_from_api()
        elif choice == "15":
            print("Paste CSV data below (end with an empty line):")
            csv_lines = []
            while True:
                line = input()
                if not line:
                    break
                csv_lines.append(line)
            csv_text = "\n".join(csv_lines)
            bulk_import_movies(conn, csv_text)
        elif choice == "16":
            print("Exiting...")
            conn.close()
            break
        else:
            print("Invalid choice. Please try again.")
        
        input("\nPress Enter to continue...")

if __name__ == "__main__":
    main()