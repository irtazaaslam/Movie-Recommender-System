import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Flatten, Dense, Concatenate
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Loading data
def load_data():
    movies_df = pd.read_csv('data/movies.csv')
    ratings_df = pd.read_csv('data/ratings.csv')
    merged_df = pd.merge(ratings_df, movies_df, on="movieId")
    return movies_df, ratings_df, merged_df

# Splitting data
def split_user_ratings(df):
    train_list, test_list = [], []
    for uid, group in df.groupby('userId'):
        train, test = train_test_split(group, test_size=0.5, random_state=42)
        train_list.append(train)
        test_list.append(test)
    return pd.concat(train_list), pd.concat(test_list)

movies_df, ratings_df, merged_df = load_data()
train_df, test_df = split_user_ratings(merged_df)

# Define the Deep Learning model
def create_dl_model(num_users, num_movies):
    user_input = Input(shape=(1,))
    movie_input = Input(shape=(1,))
    user_embedding = Embedding(output_dim=50, input_dim=num_users, input_length=1)(user_input)
    movie_embedding = Embedding(output_dim=50, input_dim=num_movies, input_length=1)(movie_input)
    user_vector = Flatten()(user_embedding)
    movie_vector = Flatten()(movie_embedding)
    concat = Concatenate()([user_vector, movie_vector])
    dense = Dense(128, activation='relu')(concat)
    output = Dense(1)(dense)
    model = Model([user_input, movie_input], output)
    model.compile(optimizer=Adam(0.002), loss='mean_squared_error')
    return model


# Encode user IDs and movie IDs
all_user_ids = np.concatenate([ratings_df['userId'].values, test_df['userId'].values])
all_movie_ids = np.concatenate([ratings_df['movieId'].values, test_df['movieId'].values])

user_encoder = LabelEncoder().fit(all_user_ids)
movie_encoder = LabelEncoder().fit(all_movie_ids)

# Now, apply the transformation
train_df['userId'] = user_encoder.transform(train_df['userId'])
train_df['movieId'] = movie_encoder.transform(train_df['movieId'])
test_df['userId'] = user_encoder.transform(test_df['userId'])
test_df['movieId'] = movie_encoder.transform(test_df['movieId'])

# Model training
num_users = len(user_encoder.classes_)
num_movies = len(movie_encoder.classes_)
model = create_dl_model(num_users, num_movies)
train_user_ids = np.array(train_df['userId'])
train_movie_ids = np.array(train_df['movieId'])
train_ratings = np.array(train_df['rating'])
model.fit([train_user_ids, train_movie_ids], train_ratings, batch_size=32, epochs=5, verbose=2)

all_movie_ids = pd.concat([ratings_df['movieId'], test_df['movieId']]).unique()
movie_encoder = LabelEncoder().fit(all_movie_ids)


def evaluate_model_performance(actual_ratings, predicted_ratings, threshold=4.0):
    from sklearn.metrics import precision_score, recall_score, confusion_matrix
    
    actual_binary = (actual_ratings >= threshold).astype(int)
    predicted_binary = (predicted_ratings >= threshold).astype(int)
    
    precision = precision_score(actual_binary, predicted_binary, zero_division=0)
    recall = recall_score(actual_binary, predicted_binary, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(actual_binary, predicted_binary).ravel()
    
    return precision, recall, fp, fn

@st.cache(allow_output_mutation=True)
def generate_recommendations(user_id, n_recommendations=5):
    user_idx = np.where(user_encoder.classes_ == user_id)[0][0]  # Convert user_id to model's encoding
    all_movie_ids = np.array(list(range(num_movies)))  # All movie IDs in the model's encoding
    predicted_ratings = model.predict([np.array([user_idx]*num_movies), all_movie_ids]).flatten()
    top_movie_indices = predicted_ratings.argsort()[-n_recommendations:][::-1]
    top_movie_ids = movie_encoder.inverse_transform(top_movie_indices)
    recommendations = movies_df[movies_df['movieId'].isin(top_movie_ids)]
    return recommendations

def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Data Overview", "Feature Enhancement", "Test-Train Split Overview", "Recommendation Abstract", "Recommendation Demo"])

    if page == "Data Overview":
        data_overview()
    elif page == "Feature Enhancement":
        feature_enhancement()
    elif page == "Test-Train Split Overview":
        test_train_split_overview()
    elif page == "Recommendation Abstract":
        recommendation_abstract()
    elif page == "Recommendation Demo":
        recommendation_demo()

def data_overview():
    st.title('Data Overview')
    st.write(movies_df.head())
    st.write('Total movies:', movies_df.shape[0])
    st.write(ratings_df.head())
    st.write('Total ratings:', ratings_df.shape[0])

def feature_enhancement():
    st.title('Feature Enhancement')
    st.markdown('''
    - **Data Merging:** Merged movie titles with ratings for easier interpretation.
    - **User and Movie IDs Encoding:** For the deep learning model, user and movie IDs are encoded to a continuous range.
    ''')

def test_train_split_overview():
    st.title('Test-Train Split Overview')
    st.write('We have splited our dataset evenly on the basis of user IDs by defining the following function')
    st.code('''
    for uid, group in df.groupby('userId'):
        train, test = train_test_split(group, test_size=0.5, random_state=42)
    ''', language='python')

def recommendation_abstract():
    st.title('Recommendation Abstract')
    st.write("""
    We created 2 models for collaborative filtering
             **A Machine Learning Model Using Cosine Similarity** &
             **A Deep Learning Model**.""")
    st.write('The deep learning model was chosen for its superior ability to predict user preferences based on historical data. The model uses embedding layers for users and movies to capture the nuances of user preferences and movie characteristics.')
    
    # Display ML Model Performance
    st.subheader("Model Evaluation:")
    st.image("images/ML Model Performance.png",)

    # Display DL Model Performance

    st.image("images/DL Model Performance.png")

def recommendation_demo():
    st.title("Movie Recommendation System")
    user_id_input = st.number_input("Enter User ID", value=1, min_value=1, max_value=int(ratings_df['userId'].max()), step=1)
    
    if st.button("Show Recommendations"):
        user_ratings = train_df[train_df['userId'] == user_id_input]
        st.subheader("Movies rated by the user (Training Data):")
        st.write(user_ratings.merge(movies_df, on='movieId'))
    
        recommendations = generate_recommendations(user_id_input)
        st.subheader("Top 5 movie recommendations:")
        st.table(recommendations[['title']])
    
        actual_ratings, predicted_ratings = get_actual_and_predicted_ratings(user_id_input)
        if actual_ratings is not None and predicted_ratings is not None:
            precision, recall, fp, fn = evaluate_model_performance(actual_ratings, predicted_ratings)
            st.subheader("Evaluation Metrics on Test Set:")
            st.write(f"Precision: {precision:.4f}")
            st.write(f"Recall: {recall:.4f}")
            st.write(f"False Positives (FP): {fp}")
            st.write(f"False Negatives (FN): {fn}")
        else:
            st.write("This user has no ratings in the test set.")



# Function to get actual and predicted ratings for the user in the test set
def get_actual_and_predicted_ratings(user_id):
    user_test_ratings = test_df[test_df['userId'] == user_id]
    if user_test_ratings.empty:
        return None, None  # User has no ratings in the test set
    
    # Prepare model inputs
    user_idx = np.where(user_encoder.classes_ == user_id)[0][0]
    movie_indices = np.array([movie_encoder.transform([movie_id])[0] for movie_id in user_test_ratings['movieId']])
    
    # Predict ratings
    predicted_ratings = model.predict([np.array([user_idx] * len(movie_indices)), movie_indices]).flatten()
    
    return user_test_ratings['rating'].values, predicted_ratings

if __name__ == "__main__":
    main()
