import pandas as pd
import numpy as np
import ast
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import ndcg_score
from sklearn.metrics.pairwise import cosine_similarity
import torch
import torch.nn as nn
import torch.optim as optim

# Define Architecture
class Autoencoder(nn.Module):
        def __init__(self, input_dim, encoding_dim=32):
            super(Autoencoder, self).__init__()
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.ReLU(),
                nn.Linear(64, encoding_dim)
            )
            self.decoder = nn.Sequential(
                nn.ReLU(),
                nn.Linear(encoding_dim, 64),
                nn.ReLU(),
                nn.Linear(64, input_dim),
                nn.Sigmoid()
            )

        def forward(self, x):
            encoded = self.encoder(x)
            decoded = self.decoder(encoded)
            return decoded

def train_and_evaluate():
    """
    Run training and evaluation processes
    """
    # Load data
    df = pd.read_csv("../data/processed/auto_events.csv")
    user_ratings_df = pd.read_csv("../data/processed/sample_user_ratings.csv")
    df["Tags"] = df["Tags"].apply(ast.literal_eval)

    # Handle tag format according to dataset
    mlb = MultiLabelBinarizer()
    tags_encoded = mlb.fit_transform(df["Tags"])

    numeric_features = df[["Price Level", "Min Age", "Max Age", "Min Group Size", "Max Group Size", "Event Duration"]].fillna(0)
    scaler = StandardScaler()
    numeric_scaled = scaler.fit_transform(numeric_features)

    features = np.hstack([tags_encoded, numeric_scaled])
    activity_names = df["Activity Name"].tolist()
    activity_ids = df["Activity ID"].tolist()

    X_train, X_temp = train_test_split(features, test_size=0.3, random_state=42)
    X_val, X_test = train_test_split(X_temp, test_size=0.5, random_state=42)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

    input_dim = features.shape[1]
    model = Autoencoder(input_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    epochs = 100
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, X_train_tensor)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_tensor)
            val_loss = criterion(val_outputs, X_val_tensor)

        if (epoch+1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}")

    model_path = "autoencoder_model.pt"
    torch.save(model.state_dict(), model_path)

    model.eval()
    with torch.no_grad():
        all_tensor = torch.tensor(features, dtype=torch.float32)
        embeddings = model.encoder(all_tensor).numpy()

    # Use embeddings to make rating predictions and provide NDCG rankings
    embedding_sim_matrix = cosine_similarity(embeddings)
    embedding_sim_df = pd.DataFrame(embedding_sim_matrix, index=activity_names, columns=activity_names)

    activity_id_to_name = dict(zip(activity_ids, activity_names))

    user_ratings_df["Activity Name"] = user_ratings_df["Activity ID"].map(activity_id_to_name)

    user_ndcgs = []
    for _, group in user_ratings_df.groupby("User ID"):
        group = group[group["Activity Name"].isin(embedding_sim_df.columns)]
        if group.shape[0] < 2:
            continue
        item_names = group["Activity Name"].tolist()
        true_ratings = group["Rating"].values.reshape(1, -1)
        reference = item_names[0]
        predicted_scores = embedding_sim_df.loc[reference, item_names].values.reshape(1, -1)
        ndcg = ndcg_score(true_ratings, predicted_scores)
        user_ndcgs.append(ndcg)

    print("Final Average NDCG:", np.mean(user_ndcgs))

    with torch.no_grad():
        test_outputs = model(X_test_tensor)
        test_loss = criterion(test_outputs, X_test_tensor)

    print(f"Test Reconstruction Loss: {test_loss.item():.4f}")

def recommend_for_user(user_id, top_n=5):
    """
    Sample inference function to showcase how predictions are made
    """
    df = pd.read_csv("../data/processed/auto_events.csv")
    user_ratings_df = pd.read_csv("../data/processed/sample_user_ratings.csv")
    df["Tags"] = df["Tags"].apply(ast.literal_eval)

    mlb = MultiLabelBinarizer()
    tags_encoded = mlb.fit_transform(df["Tags"])

    numeric_features = df[["Price Level", "Min Age", "Max Age", "Min Group Size", "Max Group Size", "Event Duration"]].fillna(0)
    scaler = StandardScaler()
    numeric_scaled = scaler.fit_transform(numeric_features)

    features = np.hstack([tags_encoded, numeric_scaled])
    activity_names = df["Activity Name"].tolist()
    activity_ids = df["Activity ID"].tolist()
    activity_id_to_name = dict(zip(activity_ids, activity_names))
    activity_name_to_index = {name: i for i, name in enumerate(activity_names)}

    user_ratings_df["Activity Name"] = user_ratings_df["Activity ID"].map(activity_id_to_name)

    model = Autoencoder(features.shape[1])
    model.load_state_dict(torch.load("autoencoder_model.pt"))
    model.eval()
    with torch.no_grad():
        all_tensor = torch.tensor(features, dtype=torch.float32)
        embeddings = model.encoder(all_tensor).numpy()

    embedding_sim_matrix = cosine_similarity(embeddings)
    embedding_sim_df = pd.DataFrame(embedding_sim_matrix, index=activity_names, columns=activity_names)

    group = user_ratings_df[user_ratings_df["User ID"] == user_id]
    group = group[group["Activity Name"].isin(embedding_sim_df.columns)]

    if group.empty:
        return f"No data for user {user_id}."

    liked = group[group["Rating"] >= 4]["Activity Name"]
    if liked.empty:
        return f"No high-rated activities for user {user_id}."

    liked_indices = [activity_name_to_index[name] for name in liked]
    user_embedding = np.mean(embeddings[liked_indices], axis=0).reshape(1, -1)

    similarities = cosine_similarity(user_embedding, embeddings).flatten()
    sim_scores = list(zip(activity_names, similarities))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    already_rated = set(group["Activity Name"])
    recommendations = [(name, score) for name, score in sim_scores if name not in already_rated]

    return recommendations[:top_n]

if __name__ == "__main__":
    train_and_evaluate()

    print("\nRecommendations for user 1:")
    for rec in recommend_for_user(1):
        print(f"{rec[0]} (score: {rec[1]:.3f})")
