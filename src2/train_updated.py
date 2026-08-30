import os
import json
import random
import hashlib
import subprocess

import numpy as np
import mlflow
import mlflow.sklearn

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import log_loss, accuracy_score
from mlflow.models import ModelSignature
from mlflow.types.schema import Schema, TensorSpec


SEED = 42

LEARNING_RATE = 0.001
BATCH_SIZE = 64
EPOCHS = 3

DATA_PATH = "data/mnist.npz"

EXPERIMENT_NAME = "Reproducibility_Capstone"

TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://127.0.0.1:5000"
)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def get_git_commit():
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL
        ).decode().strip()

        return commit

    except Exception:
        return "unknown"


def get_file_hash(filepath):
    sha256 = hashlib.sha256()

    with open(filepath, "rb") as file:
        while True:
            chunk = file.read(8192)

            if not chunk:
                break

            sha256.update(chunk)

    return sha256.hexdigest()


def load_data():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}\n"
            "Run DVC checkout first."
        )

    data = np.load(DATA_PATH)

    X_train = data["X_train"].astype(np.float32)
    y_train = data["y_train"].astype(np.int64)

    X_test = data["X_test"].astype(np.float32)
    y_test = data["y_test"].astype(np.int64)

    # FIX 1 ---------------------------------------------------------------
    # MLPClassifier requires 2D input of shape (n_samples, n_features), but an
    # earlier version of the dataset was stored as (N, 28, 28) and produced:
    #   ValueError: Found array with dim 3, while dim <= 2 is required
    #
    # prepare_data_updated.py now writes flat (N, 784) arrays, so this reshape
    # is normally a no-op. It is kept as a guard so the script cannot break
    # again if an older .npz is checked out from DVC.
    # ---------------------------------------------------------------------
    if X_train.ndim > 2:
        X_train = X_train.reshape(X_train.shape[0], -1)

    if X_test.ndim > 2:
        X_test = X_test.reshape(X_test.shape[0], -1)

    print(f"X_train: {X_train.shape}   X_test: {X_test.shape}")

    return X_train, y_train, X_test, y_test


def main():

    print("=" * 60)
    print("MNIST REPRODUCIBILITY CAPSTONE")
    print("=" * 60)

    set_seed(SEED)

    print("Device: CPU (scikit-learn)")

    git_commit = get_git_commit()

    print(f"Git commit: {git_commit}")

    dataset_hash = get_file_hash(DATA_PATH)

    print(f"Dataset SHA256: {dataset_hash}")

    (
        X_train,
        y_train,
        X_test,
        y_test
    ) = load_data()

    # Scikit-learn MLPClassifier replaces the PyTorch MLP.
    model = MLPClassifier(
        hidden_layer_sizes=(128,),
        activation="relu",
        solver="adam",
        learning_rate_init=LEARNING_RATE,
        batch_size=BATCH_SIZE,
        max_iter=EPOCHS,
        random_state=SEED,
        shuffle=True,
        verbose=False
    )

    mlflow.set_tracking_uri(
        TRACKING_URI
    )

    mlflow.set_experiment(
        EXPERIMENT_NAME
    )

    print(
        f"MLflow URI: {TRACKING_URI}"
    )

    with mlflow.start_run(
        run_name="mnist_mlp_reproducibility"
    ) as run:

        mlflow.log_params({
            "learning_rate": LEARNING_RATE,
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "seed": SEED,
            "model": "MLPClassifier",
            "hidden_layer_1": 128,
            "input_size": X_train.shape[1],
            "output_size": 10
        })

        mlflow.set_tags({
            "git_commit": git_commit,
            "dataset_sha256": dataset_hash,
            "dataset_versioning": "DVC",
            "partner_role": "Partner_A",
            "framework": "scikit-learn"
        })

        print("\nTraining MLP...")

        # MLPClassifier performs the complete training.
        model.fit(X_train, y_train)

        # Calculate final metrics on training and test data.
        train_probabilities = model.predict_proba(X_train)
        test_probabilities = model.predict_proba(X_test)

        train_predictions = model.predict(X_train)
        test_predictions = model.predict(X_test)

        train_loss = log_loss(
            y_train,
            train_probabilities,
            labels=np.arange(10)
        )

        test_loss = log_loss(
            y_test,
            test_probabilities,
            labels=np.arange(10)
        )

        train_accuracy = accuracy_score(
            y_train,
            train_predictions
        )

        test_accuracy = accuracy_score(
            y_test,
            test_predictions
        )

        # Log the final metrics.
        mlflow.log_metrics({
            "train_loss": train_loss,
            "test_loss": test_loss,
            "train_accuracy": train_accuracy,
            "test_accuracy": test_accuracy,
            "final_test_accuracy": test_accuracy,
            "final_test_loss": test_loss
        })

        print(
            f"Train Loss: {train_loss:.6f} | "
            f"Test Loss: {test_loss:.6f} | "
            f"Test Accuracy: {test_accuracy:.6f}"
        )

        os.makedirs(
            "artifacts",
            exist_ok=True
        )

        results = {
            "run_id": run.info.run_id,
            "seed": SEED,
            "git_commit": git_commit,
            "dataset_sha256": dataset_hash,
            "final_test_accuracy": test_accuracy,
            "final_test_loss": test_loss,
            "learning_rate": LEARNING_RATE,
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS
        }

        artifact_path = (
            "artifacts/results.json"
        )

        with open(
            artifact_path,
            "w"
        ) as file:

            json.dump(
                results,
                file,
                indent=4
            )

        # FIX 2 -------------------------------------------------------------
        # Artifact upload used to abort the whole run with
        #   PermissionError: [Errno 13] Permission denied: '/home/<user>'
        # when a REMOTE client logged to this server. The cause is server-side:
        # the tracking server was started without --serve-artifacts, so it
        # hands clients an absolute path on ITS filesystem, which the remote
        # client then tries to create on its own machine.
        #
        # The proper fix is to start the server with:
        #   mlflow server --backend-store-uri sqlite:///mlflow.db \
        #       --artifacts-destination ./mlartifacts --serve-artifacts \
        #       --host 0.0.0.0 --port 5000
        # and to run fix_artifact_paths.py once for pre-existing runs.
        #
        # These try/except guards mean a misconfigured server degrades to
        # "metrics logged, artifacts skipped" instead of losing the whole run.
        # -------------------------------------------------------------------
        print(f"Artifact URI: {run.info.artifact_uri}")

        try:
            mlflow.log_artifact(artifact_path)
            print("Logged results.json")

        except Exception as error:
            print(
                f"WARNING: could not log results.json ({type(error).__name__}: {error})"
            )
            print(
                "  The tracking server is likely missing --serve-artifacts. "
                "Params and metrics are unaffected."
            )

        # MLflow scikit-learn model signature.
        input_example = X_test[:1]

        signature = ModelSignature(
            inputs=Schema(
                [
                    TensorSpec(
                        np.dtype(np.float32),
                        (-1, X_test.shape[1])
                    )
                ]
            ),
            outputs=Schema(
                [
                    TensorSpec(
                        np.dtype(np.int64),
                        (-1,)
                    )
                ]
            )
        )

        try:
            mlflow.sklearn.log_model(
                model,
                name="model",
                input_example=input_example,
                signature=signature,
                skops_trusted_types=[
                    "sklearn.neural_network._stochastic_optimizers.AdamOptimizer"
                ]
            )
            print("Logged model artifact")

        except Exception as error:
            print(
                f"WARNING: could not log the model "
                f"({type(error).__name__}: {error})"
            )
            print(
                "  Same cause as above: the server needs --serve-artifacts."
            )

        print("\n" + "=" * 60)
        print("TRAINING COMPLETED SUCCESSFULLY")
        print("=" * 60)

        print(f"Run ID: {run.info.run_id}")
        print(f"Final Test Accuracy: {test_accuracy:.6f}")
        print(f"Git Commit: {git_commit}")
        print(f"Dataset Hash: {dataset_hash}")
        print("=" * 60)

        with open(
            "artifacts/latest_run_id.txt",
            "w"
        ) as file:

            file.write(
                run.info.run_id
            )


if __name__ == "__main__":
    main()
