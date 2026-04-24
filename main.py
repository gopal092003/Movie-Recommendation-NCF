import argparse
import tensorflow as tf

from src.utils.helpers import load_config, save_pickle
from src.data.loader import load_raw_ratings
from src.data.preprocessing import preprocess_ratings, train_test_split_userwise
from src.models.ncf import build_ncf_model
from src.training.trainer import train_model
from src.evaluation.evaluator import evaluate_model
from src.inference.recommender import recommend_movies


def main(mode):
    config = load_config()

    raw = load_raw_ratings(config["data"]["path"])

    ratings, num_users, num_movies, user_enc, movie_enc, movie_dec, scaler = preprocess_ratings(
        raw,
        min_movie_ratings=config["data"]["min_movie_ratings"]
    )

    X_train, y_train, X_test, y_test, train_df, test_df = train_test_split_userwise(
        ratings,
        test_frac=config["data"]["test_split"]
    )

    # Save encoders
    save_pickle(user_enc, "models/user_encoder.pkl")
    save_pickle(movie_enc, "models/movie_encoder.pkl")
    save_pickle(movie_dec, "models/movie_decoder.pkl")

    if mode == "train":
        model = build_ncf_model(num_users, num_movies, config)
        train_model(model, X_train, y_train, X_test, y_test, config)

    elif mode == "evaluate":
        model = tf.keras.models.load_model(f"{config['paths']['model_dir']}/ncf_best_model.h5")
        evaluate_model(model, X_test, y_test, config)

    elif mode == "recommend":
        model = tf.keras.models.load_model(f"{config['paths']['model_dir']}/ncf_best_model.h5")

        user_id = 1
        recs = recommend_movies(user_id, model, user_enc, movie_dec, train_df, num_movies)

        print(f"Top recommendations for user {user_id}: {recs}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=["train", "evaluate", "recommend"])
    args = parser.parse_args()

    main(args.mode)