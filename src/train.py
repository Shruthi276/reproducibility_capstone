import os
import sys
import json
import random
import hashlib
import subprocess

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

import mlflow
import mlflow.pytorch


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

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



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

    X_train = torch.tensor(
        data["X_train"],
        dtype=torch.float32
    )

    y_train = torch.tensor(
        data["y_train"],
        dtype=torch.long
    )

    X_test = torch.tensor(
        data["X_test"],
        dtype=torch.float32
    )

    y_test = torch.tensor(
        data["y_test"],
        dtype=torch.long
    )

    return X_train, y_train, X_test, y_test



class MLP(nn.Module):

    def __init__(self):
        super().__init__()

        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(
            28 * 28,
            128
        )

        self.relu = nn.ReLU()

        self.fc2 = nn.Linear(
            128,
            10
        )

    def forward(self, x):

        x = self.flatten(x)

        x = self.fc1(x)

        x = self.relu(x)

        x = self.fc2(x)

        return x



def train_one_epoch(
    model,
    train_loader,
    criterion,
    optimizer,
    device
):

    model.train()

    total_loss = 0.0
    total_samples = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        batch_size = labels.size(0)

        total_loss += loss.item() * batch_size
        total_samples += batch_size

    average_loss = total_loss / total_samples

    return average_loss



def evaluate(
    model,
    test_loader,
    criterion,
    device
):

    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            batch_size = labels.size(0)

            total_loss += loss.item() * batch_size

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            correct += (
                predictions == labels
            ).sum().item()

            total += batch_size

    test_loss = total_loss / total
    test_accuracy = correct / total

    return test_loss, test_accuracy



def main():

    print("=" * 60)
    print("MNIST REPRODUCIBILITY CAPSTONE")
    print("=" * 60)



    set_seed(SEED)


    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Device: {device}")


    git_commit = get_git_commit()

    print(f"Git commit: {git_commit}")


    dataset_hash = get_file_hash(
        DATA_PATH
    )

    print(
        f"Dataset SHA256: {dataset_hash}"
    )


    (
        X_train,
        y_train,
        X_test,
        y_test
    ) = load_data()

    train_dataset = TensorDataset(
        X_train,
        y_train
    )

    test_dataset = TensorDataset(
        X_test,
        y_test
    )

    generator = torch.Generator()

    generator.manual_seed(SEED)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=generator,
        num_workers=0
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    model = MLP().to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
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

            "model": "MLP",

            "hidden_layer_1": 128,

            "input_size": 784,

            "output_size": 10

        })

 

        mlflow.set_tags({

            "git_commit": git_commit,

            "dataset_sha256": dataset_hash,

            "dataset_versioning": "DVC",

            "partner_role": "Partner_A"

        })



        for epoch in range(EPOCHS):

            train_loss = train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                device
            )

            test_loss, test_accuracy = evaluate(
                model,
                test_loader,
                criterion,
                device
            )

            # Log metrics for every epoch
            mlflow.log_metrics({

                "train_loss": train_loss,

                "test_loss": test_loss,

                "test_accuracy": test_accuracy

            }, step=epoch)

            print(
                f"Epoch {epoch + 1}/{EPOCHS} | "
                f"Train Loss: {train_loss:.6f} | "
                f"Test Loss: {test_loss:.6f} | "
                f"Test Accuracy: {test_accuracy:.6f}"
            )



        mlflow.log_metric(
            "final_test_accuracy",
            test_accuracy
        )

        mlflow.log_metric(
            "final_test_loss",
            test_loss
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

        mlflow.log_artifact(
            artifact_path
        )


        mlflow.pytorch.log_model(
            model,
            name="model"
        )



        print("\n" + "=" * 60)

        print(
            "TRAINING COMPLETED SUCCESSFULLY"
        )

        print("=" * 60)

        print(
            f"Run ID: {run.info.run_id}"
        )

        print(
            f"Final Test Accuracy: "
            f"{test_accuracy:.6f}"
        )

        print(
            f"Git Commit: {git_commit}"
        )

        print(
            f"Dataset Hash: {dataset_hash}"
        )

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
