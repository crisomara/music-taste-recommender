"""Re-runs notebook 03's ALS training + evaluation, logged as a real MLflow run.

This is not a new model — it reproduces exactly what notebooks/03_collaborative.ipynb
already did, wrapped in mlflow.start_run() so the hyperparameters, metrics, and the
fitted model itself are tracked and inspectable via `mlflow ui`, rather than only living
as printed notebook output.

Usage:
    python scripts/train_als_with_mlflow.py
    mlflow ui   # then open http://localhost:5000
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import mlflow
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.collaborative import ALSRecommender
from src.evaluation import evaluate_model
from src.splitting import held_out_as_dict, temporal_split

INTERACTIONS_DIR = PROJECT_ROOT / "data" / "raw" / "interactions"

FACTORS = 64
REGULARIZATION = 0.05
ITERATIONS = 20
RANDOM_STATE = 42
K = 10


def main() -> None:
    mlflow.set_experiment("music-taste-recommender-als")

    with mlflow.start_run():
        mlflow.log_params({
            "factors": FACTORS,
            "regularization": REGULARIZATION,
            "iterations": ITERATIONS,
            "random_state": RANDOM_STATE,
            "k": K,
        })

        print("Loading interactions...")
        interactions = pd.read_parquet(INTERACTIONS_DIR / "lastfm_1k_train.parquet")

        train, test = temporal_split(
            interactions, user_col="user_id", item_col="join_key",
            timestamp_col="timestamp", test_frac=0.2, min_events_per_user=5,
        )
        held_out = held_out_as_dict(test, "user_id", "join_key")
        mlflow.log_metric("train_events", len(train))
        mlflow.log_metric("test_events", len(test))
        mlflow.log_metric("n_users", train["user_id"].nunique())

        print("Fitting ALS...")
        als = ALSRecommender(
            factors=FACTORS, regularization=REGULARIZATION,
            iterations=ITERATIONS, random_state=RANDOM_STATE,
        )
        als.fit(train, user_col="user_id", item_col="join_key")

        print("Evaluating...")

        def als_recommend(user_id, k=K):
            return als.recommend(user_id, k=k)

        test_users = list(held_out.keys())
        result = evaluate_model(als_recommend, test_users, held_out, k=K)
        mlflow.log_metric("precision_at_10", result["precision_at_k"])
        mlflow.log_metric("recall_at_10", result["recall_at_k"])
        mlflow.log_metric("n_users_evaluated", result["n_users_evaluated"])

        print(f"precision@10: {result['precision_at_k']:.4f}")
        print(f"recall@10:    {result['recall_at_k']:.4f}")

        model_path = "als_model_mlflow.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(als, f)
        mlflow.log_artifact(model_path)
        Path(model_path).unlink()

        print("Run logged. Launch `mlflow ui` to inspect it.")


if __name__ == "__main__":
    main()
