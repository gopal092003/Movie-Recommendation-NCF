import numpy as np

def recommend_movies(user_id, model, user2user_encoded, movie_encoded2movie, train_df, num_movies, top_n=10):
    if user_id not in user2user_encoded:
        raise ValueError("User not found in training data")

    user_enc = user2user_encoded[user_id]

    all_movie_ids = np.arange(num_movies)
    movies_rated = train_df[train_df['user'] == user_enc]['movie'].values

    movies_to_predict = np.setdiff1d(all_movie_ids, movies_rated)

    user_array = np.full(len(movies_to_predict), user_enc)

    preds = model.predict([user_array, movies_to_predict], verbose=0).flatten()

    top_indices = movies_to_predict[np.argsort(preds)[::-1][:top_n]]

    return [movie_encoded2movie[i] for i in top_indices]