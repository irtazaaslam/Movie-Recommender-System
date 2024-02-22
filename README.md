# Movie Recommendation System
For our project we have created a movie recommender system and a sleamit app using a kaggle dataset.

This repository contains a machine learning project aimed at building a movie recommendation system. The system utilizes a deep learning model to predict user preferences based on historical data and suggests movies that users might like. The project is implemented in Python, leveraging TensorFlow for the deep learning model and Streamlit for the interactive web application.

Streamlit App Link : https://movie-recommender-system-deep-learning-app.streamlit.app/

<img width="1280" alt="image" src="https://github.com/irtazaaslam/Movie-Recommender-System/assets/122581891/7a27f236-7e8e-44d0-ae54-bce091e8bb7b">

## Features
- Data Overview: Interactive exploration of the movies and ratings data.
- Feature Enhancement: Details on the preprocessing steps and feature engineering.
- Test-Train Split Overview: Explanation of the methodology behind splitting the dataset.
- Recommendation Abstract: The rationale behind choosing the model and its performance overview.
- Recommendation Demo: A live demo where users can input a User ID to receive movie recommendations and see the model's prediction accuracy.

## Data
The project uses the MovieLens dataset, which includes movie information and user ratings. Data files should be placed in the data/ directory:

- movies.csv: Movie information including movie IDs, titles, and genres.
- ratings.csv: User ratings for movies including user IDs, movie IDs, and rating values.

## Model
The recommendation model is a deep learning model built with TensorFlow. It uses embedding layers to learn latent factors for users and movies based on user rating history. The model's architecture includes:

- Input Layers: Separate input layers for users and movies.
- Embedding Layers: To learn user and movie embeddings.
- Flatten and Concatenate: To combine user and movie embeddings.
- Dense Layers: Including a ReLU-activated layer followed by an output layer to predict ratings.

## Evaluation
The model's performance is evaluated using precision, recall, and confusion matrix metrics based on a threshold
