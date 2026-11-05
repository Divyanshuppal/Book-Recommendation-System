# Book Recommendation System Using Machine Learning

A beginner-friendly college project that recommends books similar to a selected book.

## Machine Learning method
- Content-Based Filtering
- TF-IDF text vectorization
- Cosine similarity (computed efficiently with `linear_kernel`)

The model uses the book title, author and publisher because the standard Book-Crossing Books.csv file does not contain a reliable genre/description field.

## Folder structure

```text
Book_Recommendation_System/
├── app.py
├── recommender.py
├── explore_data.py
├── requirements.txt
├── README.md
└── data/
    ├── Books.csv
    ├── Ratings.csv
    └── Users.csv
```

## Setup on Windows

1. Install Python and VS Code.
2. Open this folder in VS Code.
3. Open Terminal > New Terminal.
4. Install packages:

```bash
pip install -r requirements.txt
```

5. Download the Book-Crossing dataset and place `Books.csv`, `Ratings.csv`, and `Users.csv` inside the `data` folder.
6. Explore the dataset:

```bash
python explore_data.py
```

7. Start the website:

```bash
streamlit run app.py
```

Streamlit will show a local address in the terminal. Open that address in your browser.

## How it works

1. Load the book CSV file.
2. Clean missing values and duplicate ISBN entries.
3. Combine title, author and publisher into text.
4. Use TF-IDF to convert text into numerical vectors.
5. Compare the selected book with other books using cosine similarity.
6. Display the top matching books.

## Important limitations

- Recommendations depend on available title, author and publisher information.
- This version is content-based; it does not learn personal taste from user ratings.
- New books with very little metadata may receive weak recommendations.
- Similarity score is not the same as a user rating or prediction of whether someone will like a book.

## Viva explanation

"Our project is a Book Recommendation System using Machine Learning. It uses content-based filtering. First, we load book information from a CSV dataset and clean it using Pandas. Then TF-IDF converts the text information into numerical vectors. Cosine similarity compares the selected book with other books. Finally, the system displays the most similar books through a Streamlit web application."
