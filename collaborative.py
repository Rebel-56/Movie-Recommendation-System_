"""Item-based collaborative filtering on the MovieLens dataset.

Download MovieLens "ml-latest-small" from https://grouplens.org/datasets/movielens/
and unzip it so that these files exist:

    data/ml-latest-small/ratings.csv
    data/ml-latest-small/movies.csv

Run:
    python src/collaborative.py "Toy Story (1995)"
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

ML_DIR = Path(__file__).resolve().parents[1] / "data" / "ml-latest-small"


class CollaborativeRecommender:
    def __init__(self, ml_dir: str | Path = ML_DIR, min_ratings: int = 20):
        ml_dir = Path(ml_dir)
        ratings = pd.read_csv(ml_dir / "ratings.csv")
        self.movies = pd.read_csv(ml_dir / "movies.csv")

        # keep movies with enough ratings so similarities are meaningful
        counts = ratings.groupby("movieId")["rating"].count()
        popular = counts[counts >= min_ratings].index
        ratings = ratings[ratings["movieId"].isin(popular)]

        # user x movie matrix, centred per user to remove "harsh vs generous" bias
        matrix = ratings.pivot_table(index="userId", columns="movieId", values="rating")
        matrix = matrix.sub(matrix.mean(axis=1), axis=0).fillna(0)

        sim = cosine_similarity(matrix.T)
        self.similarity = pd.DataFrame(sim, index=matrix.columns, columns=matrix.columns)

        self.id_to_title = self.movies.set_index("movieId")["title"]
        self.title_to_id = pd.Series(
            self.movies["movieId"].values, index=self.movies["title"].str.lower()
        )

    def recommend(self, title: str, top_n: int = 10) -> pd.DataFrame:
        key = title.strip().lower()
        if key not in self.title_to_id:
            raise KeyError(f"'{title}' not found. Use the MovieLens format, e.g. 'Toy Story (1995)'.")
        movie_id = self.title_to_id[key]
        if movie_id not in self.similarity.index:
            raise KeyError(f"'{title}' has too few ratings to compute similarity.")

        scores = self.similarity[movie_id].drop(movie_id).sort_values(ascending=False).head(top_n)
        return pd.DataFrame(
            {"title": scores.index.map(self.id_to_title), "similarity": scores.values.round(3)}
        )


if __name__ == "__main__":
    movie = sys.argv[1] if len(sys.argv) > 1 else "Toy Story (1995)"
    print(CollaborativeRecommender().recommend(movie).to_string(index=False))
