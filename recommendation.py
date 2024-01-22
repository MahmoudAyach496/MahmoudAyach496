## This is code is the one I used in my recommendation blog
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder

# Load the datasets
movies = pd.read_csv('movies.csv')  # Contains movie information
ratings = pd.read_csv('ratings.csv')  # Contains user ratings
users = pd.read_csv('users.csv')  # Contains user information

# Preprocessing
# Convert movie genres to a binary format
movies['genres'] = movies['genres'].str.split('|')
movies = movies.explode('genres')
movies = pd.get_dummies(movies, columns=['genres'], prefix='', prefix_sep='').groupby('movieId').sum().reset_index()

# Merge datasets
data = ratings.merge(movies, on='movieId').merge(users, on='userId')

# Convert ratings to binary (liked/not liked)
data['liked'] = data['rating'].apply(lambda x: 1 if x >= 3.5 else 0)

# Select features and target
X = data.drop(['userId', 'movieId', 'rating', 'liked', 'timestamp'], axis=1)
y = data['liked']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Testing
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy}")

# Example prediction
sample_user = X_test.iloc[0].values.reshape(1, -1)
predicted_like = model.predict(sample_user)
print(f"Predicted Like (1: Yes, 0: No): {predicted_like[0]}")
