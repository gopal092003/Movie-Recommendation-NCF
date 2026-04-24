import numpy as np
from sklearn.preprocessing import MinMaxScaler

def preprocess_ratings(ratings, min_movie_ratings):
    ratings = ratings.drop(columns=['timestamp'])

    movie_counts = ratings['movieId'].value_counts()
    ratings = ratings[ratings['movieId'].isin(movie_counts[movie_counts >= min_movie_ratings].index)]

    scaler = MinMaxScaler()
    ratings['rating'] = scaler.fit_transform(ratings[['rating']])

    user_ids = ratings['userId'].unique().tolist()
    user2user_encoded = {x: i for i, x in enumerate(user_ids)}

    movie_ids = ratings['movieId'].unique().tolist()
    movie2movie_encoded = {x: i for i, x in enumerate(movie_ids)}

    ratings['user'] = ratings['userId'].map(user2user_encoded)
    ratings['movie'] = ratings['movieId'].map(movie2movie_encoded)

    movie_encoded2movie = {x: i for i, x in movie2movie_encoded.items()}

    num_users = len(user2user_encoded)
    num_movies = len(movie2movie_encoded)

    ratings['rating'] = ratings['rating'].astype(np.float32)

    return ratings, num_users, num_movies, user2user_encoded, movie2movie_encoded, movie_encoded2movie, scaler


def train_test_split_userwise(ratings, test_frac):
    train_rows, test_rows = [], []

    for _, user_data in ratings.groupby('user'):
        user_data = user_data.sample(frac=1, random_state=42)

        n_items = len(user_data)
        train_size = max(1, int((1 - test_frac) * n_items))

        train_rows.append(user_data.iloc[:train_size])
        test_rows.append(user_data.iloc[train_size:])

    import pandas as pd
    train_df = pd.concat(train_rows)
    test_df = pd.concat(test_rows)

    X_train = [train_df['user'].values, train_df['movie'].values]
    y_train = train_df['rating'].values

    X_test = [test_df['user'].values, test_df['movie'].values]
    y_test = test_df['rating'].values

    return X_train, y_train, X_test, y_test, train_df, test_df