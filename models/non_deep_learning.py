
import pandas as pd
import numpy as np
import ast
import joblib
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import ndcg_score
from sklearn.metrics.pairwise import cosine_similarity
import os
from rapidfuzz import process

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(CURRENT_DIR, "../data/processed/auto_events.csv")
RATINGS_PATH = os.path.join(CURRENT_DIR, "../data/processed/sample_user_ratings.csv")
MODEL_PATH = os.path.join(CURRENT_DIR, "./knn_model.pkl")
SCALER_PATH = os.path.join(CURRENT_DIR, "./scaler.pkl")
FEATURES_PATH = os.path.join(CURRENT_DIR, "../data/processed/activity_features.csv")
ACTIVITY_MAP_PATH = os.path.join(CURRENT_DIR, "../data/processed/activity_id_to_name.csv")

def train_and_evaluate_knn():
    """
    Performs training and evaluation of KNN
    """
    df = pd.read_csv(DATA_PATH)
    user_ratings_df = pd.read_csv(RATINGS_PATH)
    df["Tags"] = df["Tags"].apply(ast.literal_eval)

    mlb = MultiLabelBinarizer()
    tags_encoded = mlb.fit_transform(df["Tags"])
    tags_df = pd.DataFrame(tags_encoded, columns=mlb.classes_)

    numeric_features = df[["Price Level", "Min Age", "Max Age", "Min Group Size", "Max Group Size", "Event Duration"]].fillna(0)
    features_df = pd.concat([tags_df, numeric_features.reset_index(drop=True)], axis=1)
    features_df["Activity ID"] = df["Activity ID"].values

    merged = user_ratings_df.merge(features_df, on="Activity ID")
    X = merged.drop(columns=["User ID", "Activity ID", "Rating"])
    y = merged["Rating"]

    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = KNeighborsRegressor(n_neighbors=5)
    model.fit(X_train_scaled, y_train)

    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    features_df.set_index("Activity ID").to_csv(FEATURES_PATH)
    df[["Activity ID", "Activity Name"]].to_csv(ACTIVITY_MAP_PATH, index=False)

    activity_id_to_name = dict(zip(df["Activity ID"], df["Activity Name"]))
    user_ratings_df["Activity Name"] = user_ratings_df["Activity ID"].map(activity_id_to_name)

    activity_features_matrix = features_df.set_index("Activity ID")
    embeddings = scaler.transform(activity_features_matrix.drop(columns=["Activity ID"], errors='ignore'))
    activity_names = [activity_id_to_name[aid] for aid in activity_features_matrix.index]

    sim_matrix = cosine_similarity(embeddings)
    sim_df = pd.DataFrame(sim_matrix, index=activity_names, columns=activity_names)

    user_ndcgs = []
    for _, group in user_ratings_df.groupby("User ID"):
        group = group[group["Activity Name"].isin(sim_df.columns)]
        if group.shape[0] < 2:
            continue
        item_names = group["Activity Name"].tolist()
        true_ratings = group["Rating"].values.reshape(1, -1)
        reference = item_names[0]
        predicted_scores = sim_df.loc[reference, item_names].values.reshape(1, -1)
        ndcg = ndcg_score(true_ratings, predicted_scores)
        user_ndcgs.append(ndcg)

    print("Training complete. Average NDCG:", np.mean(user_ndcgs))

def cold_start_recommend(profile, top_n=5):
    """
    Cold start process for KNN
    """
    activity_features = pd.read_csv(FEATURES_PATH).set_index("Activity ID")
    activity_map = pd.read_csv(ACTIVITY_MAP_PATH)
    id_to_name = dict(zip(activity_map["Activity ID"], activity_map["Activity Name"]))

    mlb = MultiLabelBinarizer(classes=profile["tags_all_possible"])
    mlb.fit([[]])
    tag_vector = mlb.transform([profile["selected_tags"]])[0]

    numeric_vector = [
        profile["price_level"],
        profile["min_age"],
        profile["max_age"],
        profile["min_group_size"],
        profile["max_group_size"],
        profile["event_duration"]
    ]
    user_vector = np.concatenate([tag_vector, numeric_vector])

    # Scale user vector and activity features
    scaler = joblib.load(SCALER_PATH)
    user_vector_scaled = scaler.transform([user_vector])
    all_scaled = scaler.transform(activity_features.values)

    # Compute cosine similarity
    similarities = cosine_similarity(user_vector_scaled, all_scaled).flatten()

    activity_ids = activity_features.index.tolist()
    recommendations = [(id_to_name[aid], similarities[i]) for i, aid in enumerate(activity_ids)]

    return sorted(recommendations, key=lambda x: x[1], reverse=True)[:top_n]

def recommend(user_id, top_n=5):
    """
    Warmed up KNN recommendations
    """
    user_ratings_df = pd.read_csv(RATINGS_PATH)
    scaler = joblib.load(SCALER_PATH)
    activity_features = pd.read_csv(FEATURES_PATH)
    activity_map = pd.read_csv(ACTIVITY_MAP_PATH)

    activity_features.reset_index(inplace=True, drop=True)
    activity_features.set_index("Activity ID", inplace=True)

    activity_map["Activity ID"] = activity_map["Activity ID"].astype(int)
    activity_id_to_name = dict(zip(activity_map["Activity ID"], activity_map["Activity Name"]))
    activity_name_to_id = {v: k for k, v in activity_id_to_name.items()}
    user_ratings_df["Activity Name"] = user_ratings_df["Activity ID"].map(activity_id_to_name)

    group = user_ratings_df[user_ratings_df["User ID"] == user_id]
    liked = group[group["Rating"] >= 0]["Activity Name"]

    liked_ids = [activity_name_to_id[name] for name in liked if name in activity_name_to_id]
    liked_vectors = activity_features.loc[liked_ids].values
    user_profile = np.mean(liked_vectors, axis=0).reshape(1, -1)

    all_vectors = activity_features.values
    all_ids = activity_features.index.tolist()
    all_names = [activity_id_to_name[aid] for aid in all_ids]

    user_profile_scaled = scaler.transform(user_profile)
    all_scaled = scaler.transform(all_vectors)

    sims = cosine_similarity(user_profile_scaled, all_scaled).flatten()
    sim_scores = list(zip(all_names, sims))
    already_seen = set(group["Activity Name"])
    recommendations = [(name, score) for name, score in sim_scores if name not in already_seen]

    return sorted(recommendations, key=lambda x: x[1], reverse=True)[:top_n]

def update_model(user_id: int, activity_name: str, rating: float):    
    """
    Retrain the model when feedback is received
    """
    df = pd.read_csv(DATA_PATH)
    activity_names = df["Activity Name"].tolist()
    match, _, _ = process.extractOne(activity_name, activity_names)
    activity_id = df[df["Activity Name"] == match]["Activity ID"].values[0]

    if os.path.exists(RATINGS_PATH):
        ratings_df = pd.read_csv(RATINGS_PATH)

    ratings_df = ratings_df[
        ~((ratings_df["User ID"] == user_id) & (ratings_df["Activity ID"] == activity_id))
    ]

    # Append new rating
    new_row = pd.DataFrame([{
        "User ID": user_id,
        "Activity ID": activity_id,
        "Rating": rating
    }])
    updated_df = pd.concat([ratings_df, new_row], ignore_index=True)
    updated_df.to_csv(RATINGS_PATH, index=False)

    refit_knn_model()

def refit_knn_model():
    """
    Refit the model helper function
    """
    df = pd.read_csv(DATA_PATH)
    ratings_df = pd.read_csv(RATINGS_PATH)
    df["Tags"] = df["Tags"].apply(eval)

    mlb = MultiLabelBinarizer()
    tags_encoded = mlb.fit_transform(df["Tags"])
    tags_df = pd.DataFrame(tags_encoded, columns=mlb.classes_)

    numeric_features = df[["Price Level", "Min Age", "Max Age", "Min Group Size", "Max Group Size", "Event Duration"]].fillna(0)
    features_df = pd.concat([tags_df, numeric_features.reset_index(drop=True)], axis=1)
    features_df["Activity ID"] = df["Activity ID"].values

    merged = ratings_df.merge(features_df, on="Activity ID")
    X = merged.drop(columns=["User ID", "Activity ID", "Rating"])
    y = merged["Rating"]

    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = KNeighborsRegressor(n_neighbors=5)
    model.fit(X_train_scaled, y_train)

    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    features_df.set_index("Activity ID").to_csv(FEATURES_PATH)
    df[["Activity ID", "Activity Name"]].to_csv(ACTIVITY_MAP_PATH, index=False)

if __name__ == "__main__":
    train_and_evaluate_knn()

    print(recommend(2))
    TAGS_ALL = [
        "Adults-Only", "Adventure", "Budget-Friendly", "Concert", "Creative", "Cultural", "Educational",
        "Entertainment", "Expensive", "Fall", "Family-Friendly", "Festival", "Free", "Game", "High",
        "Indoor", "Low", "Market", "Medium", "Meetup", "Mixed", "Outdoor", "Relaxing", "Seasonal",
        "Social", "Spring", "Summer", "Winter", "Workshop", "Year-Round"
    ]
    profile = {
        "selected_tags": ["Adventure", "Outdoor", "Budget-Friendly", "Creative"],
        "tags_all_possible": TAGS_ALL,
        "price_level": 2,
        "min_age": 18,
        "max_age": 35,
        "min_group_size": 2,
        "max_group_size": 6,
        "event_duration": 2
    }
    print(cold_start_recommend(profile=profile))