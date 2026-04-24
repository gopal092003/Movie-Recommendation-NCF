import pandas as pd

def load_raw_ratings(path):
    return pd.read_csv(path)