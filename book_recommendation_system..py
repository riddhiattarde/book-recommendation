import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Updated dataset of books with genres and descriptions
data = {
    'Book': ['The Shining', 'Pride and Prejudice', 'A Good Girl\'s Guide to Murder', 'Book Lovers', 'The Family Upstairs'],
    'Genre': ['Horror and Mystery', 'Satire', 'Mystery and YA', 'YA and Romance', 'Mystery and Thriller'],
    'Description': [
        'A chilling horror story about a haunted hotel',
        'A witty exploration of class and love in early 19th century England',
        'A gripping mystery involving a missing girl',
        'A romance set against the backdrop of the literary world',
        'A psychological thriller with dark family secrets'
    ]
}

books_df = pd.DataFrame(data)

# Combine Genre and Description into a single string for each book to vectorize
books_df['Features'] = books_df['Genre'] + " " + books_df['Description']

# Use TF-IDF Vectorizer to transform the text data into vectors
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(books_df['Features'])

# Calculate cosine similarity between books based on their features
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Function to get book recommendations
def recommend_books(book_title, cosine_sim=cosine_sim):
    # Get the index of the book that matches the title
    idx = books_df[books_df['Book'] == book_title].index[0]

    # Get the similarity scores for all books with the selected book
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the books based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the indices of the 3 most similar books (excluding the input book)
    sim_scores = sim_scores[1:4]

    # Get the book indices
    book_indices = [i[0] for i in sim_scores]

    # Return the top 3 most similar books
    return books_df['Book'].iloc[book_indices]

# Example: Get recommendations for 'The Shining'
print("Recommendations for 'The Shining':")
print(recommend_books('The Shining'))

# Example: Get recommendations for 'Pride and Prejudice'
print("\nRecommendations for 'Pride and Prejudice':")
print(recommend_books('Pride and Prejudice'))
