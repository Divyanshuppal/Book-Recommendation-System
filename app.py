import streamlit as st
import pandas as pd
from recommender import load_books, build_recommender, recommend_books

st.set_page_config(
    page_title="Book Recommendation System",
    page_icon="📚",
    layout="wide"
)

st.title("📚 Book Recommendation System")
st.write("Find books similar to your favourite books.")

# Load book dataset
@st.cache_data
def get_books():
    data = load_books("data/Books.csv")

    if data is None:
        raise ValueError("load_books() returned None")

    return data

# Load recommendation model
@st.cache_resource
def get_model(data):
    return build_recommender(data)

# Initialize books variable
books = None

try:
    books = get_books()

except FileNotFoundError:
    st.error("Books.csv not found inside the data folder.")
    st.stop()

except Exception as e:
    st.error(f"Error loading dataset: {e}")
    st.stop()

# Check whether dataset is empty
books = get_books()

if books is None or books.empty:
    st.error("Dataset is empty or not loaded.")
    st.stop()

st.success(f"Successfully loaded {len(books)} books!")

# Build ML model
try:
    model_data = get_model(books)

except Exception as e:
    st.error(f"Error building recommendation model: {e}")
    st.stop()

st.divider()

# Book selection
st.subheader("Select Your Favourite Book")

selected_book = st.selectbox(
    "Choose a book",
    books["Book-Title"].tolist()
)

number = st.slider(
    "Number of recommendations",
    3, 10, 5
)

# Recommendation button
if st.button("Recommend Books", type="primary"):

    results = recommend_books(
        selected_book,
        model_data,
        n=number
    )

    if results.empty:
        st.warning("No similar books found.")

    else:
        st.subheader("Recommended Books")

        for index, row in results.iterrows():

            st.markdown(f"### 📖 {row['Book-Title']}")

            st.write("Author:", row["Book-Author"])
            st.write("Publisher:", row["Publisher"])
            st.write(
                "Publication Year:",
                row["Year-Of-Publication"]
            )
            st.write(
                "Similarity Score:",
                round(row["Similarity"], 3)
            )

            st.divider()

# Dataset information
with st.expander("Explore Dataset"):

    st.write("First five books:")
    st.dataframe(books.head())

    st.write("Dataset shape:")
    st.write(books.shape)

    st.write("Missing values:")
    st.write(books.isnull().sum())
