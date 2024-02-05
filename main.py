import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Flatten, Dense, Concatenate
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import mean_squared_error, precision_score, recall_score, f1_score, confusion_matrix


# Loading data
def load_data():
    movies_df = pd.read_csv('movies.csv')
    ratings_df = pd.read_csv('ratings.csv')
    merged_df = pd.merge(ratings_df, movies_df, on="movieId")
    return movies_df, ratings_df, merged_df

movies_df, ratings_df, merged_df = load_data()

# Splitting data
def split_user_ratings(df):
    train_list, test_list = [], []
    for uid, group in df.groupby('userId'):
        train, test = train_test_split(group, test_size=0.5, random_state=42)
        train_list.append(train)
        test_list.append(test)
    return pd.concat(train_list), pd.concat(test_list)

train_df, test_df = split_user_ratings(merged_df)

# User-item matrix and cosine similarity
def create_similarity_matrix(df):
    user_item_matrix = df.pivot_table(index='userId', columns='title', values='rating').fillna(0)
    sparse_user_item = csr_matrix(user_item_matrix.values)
    cosine_sim = cosine_similarity(sparse_user_item)
    return user_item_matrix, cosine_sim

user_item_matrix, cosine_sim = create_similarity_matrix(train_df)

# Predict rating function adjusted for movie ID, including movie title in output
def predict_rating(user_id, movie_id, num_similar_users=5):
    if movie_id not in movies_df['movieId'].values:
        return None, "Movie ID not found"
    movie_title = movies_df[movies_df['movieId'] == movie_id]['title'].iloc[0]
    
    if movie_title not in user_item_matrix.columns:
        return None, "Movie not rated by any user in training set"
    
    user_idx = user_item_matrix.index.get_loc(user_id)
    sim_scores = cosine_sim[user_idx]
    
    top_user_indices = np.argsort(sim_scores)[::-1][1:num_similar_users+1]
    
    weighted_sum, sim_score_sum = 0, 0
    for idx in top_user_indices:
        if user_item_matrix.iloc[idx][movie_title] > 0:
            weighted_sum += sim_scores[idx] * user_item_matrix.iloc[idx][movie_title]
            sim_score_sum += sim_scores[idx]
    
    predicted_rating = weighted_sum / sim_score_sum if sim_score_sum > 0 else None
    return predicted_rating, movie_title

# Predict ratings for the test dataset, including movie titles
def predict_test_ratings(test_df, num_similar_users=5):
    predictions = []
    for index, row in test_df.iterrows():
        user_id = row['userId']
        movie_id = row['movieId']
        actual_rating = row['rating']
        predicted_rating, movie_title = predict_rating(user_id, movie_id, num_similar_users)
        
        if predicted_rating is not None:
            predictions.append({
                'userId': user_id,
                'movieId': movie_id,
                'Movie Title': movie_title,
                'Actual Rating': actual_rating,
                'Predicted Rating': predicted_rating
            })
    return pd.DataFrame(predictions)



test_predictions_ml = predict_test_ratings(test_df)  # Demonstrating with a subset for efficiency

# Adding Deep Learning Model
# Encode user IDs and movie IDs for the neural network
user_ids = ratings_df['userId'].astype('category').cat.codes.values
movie_ids = ratings_df['movieId'].astype('category').cat.codes.values

# Map user and movie IDs to a continuous range of integers
user_id_input = {id: idx for idx, id in enumerate(ratings_df['userId'].unique())}
movie_id_input = {id: idx for idx, id in enumerate(ratings_df['movieId'].unique())}

# Prepare inputs for the DL model
train_user_ids = np.array(train_df['userId'].map(user_id_input))
train_movie_ids = np.array(train_df['movieId'].map(movie_id_input))
train_ratings = np.array(train_df['rating'])

test_user_ids = np.array(test_df['userId'].map(user_id_input)[:100])  # Matching the ML model's subset
test_movie_ids = np.array(test_df['movieId'].map(movie_id_input)[:100])

# Define the Deep Learning model
def create_dl_model():
    user_input = Input(shape=(1,))
    movie_input = Input(shape=(1,))
    user_embedding = Embedding(output_dim=50, input_dim=len(user_id_input), input_length=1)(user_input)
    movie_embedding = Embedding(output_dim=50, input_dim=len(movie_id_input), input_length=1)(movie_input)
    user_vector = Flatten()(user_embedding)
    movie_vector = Flatten()(movie_embedding)
    concat = Concatenate()([user_vector, movie_vector])
    dense = Dense(128, activation='relu')(concat)
    output = Dense(1)(dense)
    model = Model([user_input, movie_input], output)
    model.compile(optimizer=Adam(0.002), loss='mean_squared_error')
    return model

# Instantiate and train the DL model
dl_model = create_dl_model()
dl_model.fit([train_user_ids, train_movie_ids], train_ratings, batch_size=32, epochs=5, verbose=2)

# Make predictions with the DL model
dl_predictions = dl_model.predict([test_user_ids, test_movie_ids])


# Correct approach to match lengths when assigning DL predictions to the DataFrame
test_predictions_subset = test_predictions_ml.head(len(dl_predictions))  # Match the subset size
test_predictions_subset['DL Predicted Rating'] = dl_predictions.flatten()

# Now, you can print or evaluate this subset
print(test_predictions_subset[['userId', 'movieId', 'Movie Title', 'Actual Rating', 'Predicted Rating', 'DL Predicted Rating']].head())


def evaluate_model_performance(actual_ratings, predicted_ratings, threshold=4.0):
    from sklearn.metrics import precision_score, recall_score, confusion_matrix
    
    # Convert continuous ratings to binary (liked/not liked)
    actual_binary = (actual_ratings >= threshold).astype(int)
    predicted_binary = (predicted_ratings >= threshold).astype(int)
    
    # Calculate metrics
    precision = precision_score(actual_binary, predicted_binary)
    recall = recall_score(actual_binary, predicted_binary)
    tn, fp, fn, tp = confusion_matrix(actual_binary, predicted_binary).ravel()
    
    # Print evaluation metrics
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'False Positives (FP): {fp}')
    print(f'False Negatives (FN): {fn}')

# Assuming test_predictions_ml contains ML model predictions
# Assuming dl_predictions contains DL model predictions

# Extract actual ratings for evaluation
actual_ratings_ml = test_predictions_ml['Actual Rating']
predicted_ratings_ml = test_predictions_ml['Predicted Rating']

# Assuming the DL model predictions (dl_predictions) are aligned with the actual ratings for the DL model
# For simplicity, let's align DL predictions with the actual ratings from the same subset used for ML predictions
actual_ratings_dl = actual_ratings_ml[:len(dl_predictions)]
predicted_ratings_dl = dl_predictions.flatten()

print("Evaluating ML Model Performance:")
evaluate_model_performance(actual_ratings_ml, predicted_ratings_ml)

print("\nEvaluating DL Model Performance:")
evaluate_model_performance(actual_ratings_dl, predicted_ratings_dl)

