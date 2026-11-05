import pandas as pd

# Read the three Book-Crossing CSV files.
# sep=None detects comma or semicolon delimiters.
books = pd.read_csv("data/Books.csv", sep=None, engine="python", encoding="latin-1", on_bad_lines="skip")
ratings = pd.read_csv("data/Ratings.csv", sep=None, engine="python", encoding="latin-1", on_bad_lines="skip")
users = pd.read_csv("data/Users.csv", sep=None, engine="python", encoding="latin-1", on_bad_lines="skip")

print("\\n--- BOOK DATA ---")
print(books.head())
print("\\n--- RATINGS DATA ---")
print(ratings.head())
print("\\n--- USERS DATA ---")
print(users.head())

print("\\n--- DATASET SHAPES ---")
print("Books:", books.shape)
print("Ratings:", ratings.shape)
print("Users:", users.shape)

print("\\n--- BOOK COLUMNS ---")
print(books.columns.tolist())

print("\\n--- MISSING VALUES IN BOOKS ---")
print(books.isnull().sum())

print("\\n--- DUPLICATE ROWS ---")
print("Books:", books.duplicated().sum())
print("Ratings:", ratings.duplicated().sum())
print("Users:", users.duplicated().sum())

print("\\n--- BOOK DATA INFORMATION ---")
books.info()
