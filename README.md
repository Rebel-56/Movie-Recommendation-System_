# 🎬 Movie Recommender System

A movie recommendation system built in Python with two approaches:

1. **Content-based filtering** – recommends movies similar to one you like, using TF-IDF on genres + plot overview and cosine similarity.
2. **Item-based collaborative filtering** – recommends movies that users with similar rating patterns liked (uses the MovieLens dataset).

A small **Streamlit** web app lets you try the content-based model interactively.

## Tech stack
Python · Pandas · NumPy · Scikit-learn · Streamlit

## Project structure
```
movie-recommender-system/
├── app.py                  # Streamlit web app
├── data/
│   └── movies_sample.csv   # 40-movie sample dataset (works out of the box)
├── src/
│   ├── recommender.py      # content-based model
│   └── collaborative.py    # collaborative filtering (MovieLens)
├── requirements.txt
└── README.md
```

## Getting started
```bash
git clone https://github.com/<your-username>/movie-recommender-system.git
cd movie-recommender-system
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Run the content-based recommender
```bash
python src/recommender.py
```
```python
from src.recommender import MovieRecommender

rec = MovieRecommender()
rec.recommend("Inception", top_n=5)
rec.search("friends road trip comedy")
```

### Run the web app
```bash
streamlit run app.py
```

### Use the full MovieLens dataset (optional)
1. Download **ml-latest-small** from https://grouplens.org/datasets/movielens/
2. Unzip into `data/ml-latest-small/`
3. Run:
```bash
python src/collaborative.py "Toy Story (1995)"
```

## How it works
**Content-based:** for every movie, genres (weighted 3x) and the plot overview are combined into one text field and converted to TF-IDF vectors. The similarity between two movies is the cosine of the angle between their vectors; the highest-scoring movies are recommended.

**Collaborative:** a user × movie ratings matrix is mean-centred per user, then cosine similarity is computed between movie columns. Movies rated similarly by the same users end up close together.

## Ideas for improvement
- Add a hybrid model combining both scores
- Add cast/director features and movie posters (TMDB API)
- Evaluate with precision@k / RMSE on a train-test split
- Deploy on Streamlit Community Cloud

## Author
Abhishek S
