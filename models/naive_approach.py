import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder
from sklearn.metrics import ndcg_score, mean_squared_error
import ast

DATA_PATH = "../data/processed/auto_events.csv"
RATINGS_PATH = "../data/processed/sample_user_ratings.csv"

tag_means = {}
mlb = None
tag_columns = None
merged_df = None

def train_and_evaluate_naive():
    global tag_means, mlb, tag_columns, merged_df

    df = pd.read_csv(DATA_PATH)
    user_ratings_df = pd.read_csv(RATINGS_PATH)
    merged_df = user_ratings_df.merge(df, on="Activity ID")
    merged_df["Tags"] = merged_df["Tags"].apply(ast.literal_eval)

    mlb = MultiLabelBinarizer()
    tags_encoded = mlb.fit_transform(merged_df["Tags"])

    tags_df = pd.DataFrame(tags_encoded, columns=mlb.classes_)
    merged_with_tags = pd.concat([merged_df[["User ID", "Activity ID", "Rating"]], tags_df], axis=1)

    tag_columns = tags_df.columns
    tag_means = {}
    for tag in tag_columns:
        tag_rows = merged_with_tags[merged_with_tags[tag] == 1]
        tag_means[tag] = tag_rows["Rating"].mean() if not tag_rows.empty else 0.0

    def predict_naive(row):
        tags = [tag for tag in tag_columns if row[tag] == 1]
        return np.mean([tag_means[tag] for tag in tags]) if tags else 3.0

    merged_with_tags["Predicted_Rating"] = merged_with_tags.apply(predict_naive, axis=1)

    mse = mean_squared_error(merged_with_tags["Rating"], merged_with_tags["Predicted_Rating"])
    print("Mean Squared Error:", mse)

    user_ndcgs = []
    for _, group in merged_with_tags.groupby("User ID"):
        if group.shape[0] < 2:
            continue
        true_relevance = group["Rating"].values.reshape(1, -1)
        predicted_scores = group["Predicted_Rating"].values.reshape(1, -1)
        ndcg = ndcg_score(true_relevance, predicted_scores)
        user_ndcgs.append(ndcg)

    average_ndcg = np.mean(user_ndcgs)
    print("Average NDCG:", average_ndcg)

def recommend_naive(user_id, top_n=5):
    global tag_means, mlb, merged_df

    if merged_df is None or mlb is None:
        raise ValueError("Model not trained. Call train_and_evaluate_naive() first.")

    df = pd.read_csv(DATA_PATH)
    user_ratings_df = pd.read_csv(RATINGS_PATH)
    df["Tags"] = df["Tags"].apply(ast.literal_eval)

    rated_ids = user_ratings_df[user_ratings_df["User ID"] == user_id]["Activity ID"].tolist()
    unrated_df = df[~df["Activity ID"].isin(rated_ids)].copy()

    tag_vectors = mlb.transform(unrated_df["Tags"])
    tag_columns = mlb.classes_

    def predict_tags(tags_vector):
        tags = [tag_columns[i] for i, val in enumerate(tags_vector) if val == 1]
        return np.mean([tag_means.get(tag, 0) for tag in tags]) if tags else 3.0

    unrated_df["Predicted_Rating"] = [predict_tags(row) for row in tag_vectors]
    recommendations = unrated_df[["Activity Name", "Predicted_Rating"]].sort_values(by="Predicted_Rating", ascending=False)

    return recommendations.head(top_n).values.tolist()

if __name__ == "__main__":
    train_and_evaluate_naive()

    print(recommend_naive(1))