"""Content-based movie recommender.

Each movie is turned into a text "soup" (genres + overview), vectorised with
TF-IDF, and compared with cosine similarity. Movies closest to the one the
user picked are returned as recommendations.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

DEFAULT_DATA = Path(__file__).resolve().parents[1] / "data" / "movies_sample.csv"


class MovieRecommender:
    def __init__(self, data_path: str | Path = DEFAULT_DATA, genre_weight: int = 3):
        """
        Parameters
        ----------
        data_path    : CSV with columns movie_id, title, year, genres, overview.
                       Genres are separated by '|'.
        genre_weight : how many times genres are repeated in the text soup, so
                       genre matches count more than random overview words.
        """
        self.movies = pd.read_csv(data_path).fillna("")
        self.movies["genres_list"] = self.movies["genres"].str.split("|")
        self.genre_weight = genre_weight

        self.movies["soup"] = self.movies.apply(self._build_soup, axis=1)

        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self.tfidf_matrix = self.vectorizer.fit_transform(self.movies["soup"])

        # case-insensitive title -> row index lookup
        self._index = pd.Series(
            self.movies.index, index=self.movies["title"].str.lower()
        ).drop_duplicates()

    def _build_soup(self, row: pd.Series) -> str:
        genres = " ".join(g.replace(" ", "").lower() for g in row["genres_list"])
        return f"{(genres + ' ') * self.genre_weight}{row['overview']}"

    # ------------------------------------------------------------------ API
    def titles(self) -> list[str]:
        return sorted(self.movies["title"].tolist())

    def recommend(self, title: str, top_n: int = 5) -> pd.DataFrame:
        """Return the top_n movies most similar to `title`."""
        key = title.strip().lower()
        if key not in self._index:
            raise KeyError(f"Movie '{title}' not found in the dataset.")

        idx = self._index[key]
        # TF-IDF rows are L2-normalised, so a dot product equals cosine similarity.
        scores = linear_kernel(self.tfidf_matrix[idx], self.tfidf_matrix).ravel()

        order = scores.argsort()[::-1]
        order = [i for i in order if i != idx][:top_n]

        result = self.movies.iloc[order][["title", "year", "genres"]].copy()
        result["similarity"] = scores[order].round(3)
        return result.reset_index(drop=True)

    def recommend_by_genre(self, genre: str, top_n: int = 5) -> pd.DataFrame:
        """Simple helper: movies that contain a genre, newest first."""
        mask = self.movies["genres_list"].apply(
            lambda gs: genre.lower() in [g.lower() for g in gs]
        )
        return (
            self.movies[mask]
            .sort_values("year", ascending=False)
            .head(top_n)[["title", "year", "genres"]]
            .reset_index(drop=True)
        )

    def search(self, query: str, top_n: int = 5) -> pd.DataFrame:
        """Free-text search: describe what you want to watch."""
        q_vec = self.vectorizer.transform([query])
        scores = linear_kernel(q_vec, self.tfidf_matrix).ravel()
        order = scores.argsort()[::-1][:top_n]
        result = self.movies.iloc[order][["title", "year", "genres"]].copy()
        result["similarity"] = scores[order].round(3)
        return result.reset_index(drop=True)


if __name__ == "__main__":
    rec = MovieRecommender()
    print("Because you liked 'Inception':\n")
    print(rec.recommend("Inception", top_n=5).to_string(index=False))
