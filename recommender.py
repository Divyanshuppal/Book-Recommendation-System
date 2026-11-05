
import os
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


def load_books(file_path):

    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Dataset not found: {file_path}"
        )

    # Read CSV file
    books = pd.read_csv(
        file_path,
        sep=None,
        engine="python",
        encoding="latin-1",
        on_bad_lines="skip"
    )

    # Clean column names
    books.columns = books.columns.str.strip()

    print("Dataset loaded successfully!")
    print("Columns:", books.columns.tolist())

    # Select required columns
    required_columns = [
        "ISBN",
        "Book-Title",
        "Book-Author",
        "Year-Of-Publication",
        "Publisher"
    ]

    for col in required_columns:
        if col not in books.columns:
            raise ValueError(
                f"Missing column: {col}. Check your CSV file."
            )

    books = books[required_columns].copy()

    # Handle missing values
    for col in ["ISBN", "Book-Title", "Book-Author", "Publisher"]:
        books[col] = books[col].fillna("").astype(str).str.strip()

    books["Year-Of-Publication"] = (
        books["Year-Of-Publication"].fillna("").astype(str)
    )

    # Remove empty titles
    books = books[books["Book-Title"] != ""]

    # Remove duplicate ISBN
    books = books.drop_duplicates(
        subset=["ISBN"],
        keep="first"
    )

    books = books.reset_index(drop=True)

    # Create combined text
    books["combined_features"] = (
        books["Book-Title"] + " " +
        books["Book-Author"] + " " +
        books["Publisher"]
    ).str.lower()

    return books


def build_recommender(books):

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_features=50000
    )

    tfidf_matrix = vectorizer.fit_transform(
        books["combined_features"]
    )

    return {
        "books": books,
        "vectorizer": vectorizer,
        "tfidf_matrix": tfidf_matrix
    }


def recommend_books(book_title, model_data, n=5):

    books = model_data["books"]
    tfidf_matrix = model_data["tfidf_matrix"]

    matches = books.index[
        books["Book-Title"].str.casefold()
        == book_title.casefold()
    ].tolist()

    if not matches:
        return pd.DataFrame()

    selected_index = matches[0]

    similarity_scores = linear_kernel(
        tfidf_matrix[selected_index:selected_index + 1],
        tfidf_matrix
    ).flatten()

    similar_indices = similarity_scores.argsort()[::-1]

    similar_indices = [
        i for i in similar_indices
        if i != selected_index
    ][:n]

    results = books.iloc[similar_indices][
        [
            "Book-Title",
            "Book-Author",
            "Publisher",
            "Year-Of-Publication"
        ]
    ].copy()

    results["Similarity"] = similarity_scores[similar_indices]

    return results.reset_index(drop=True)
