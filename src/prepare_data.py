import os
import random
import numpy as np
import torch
from torchvision.datasets import MNIST


SEED = 42
DATA_PATH = "data/mnist.npz"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main():
    set_seed(SEED)

    os.makedirs("data", exist_ok=True)

    print("Downloading/loading MNIST...")

    train_dataset = MNIST(
        root="data/raw",
        train=True,
        download=True
    )

    test_dataset = MNIST(
        root="data/raw",
        train=False,
        download=True
    )

    X_train = train_dataset.data.numpy().astype(np.float32) / 255.0
    y_train = train_dataset.targets.numpy().astype(np.int64)

    X_test = test_dataset.data.numpy().astype(np.float32) / 255.0
    y_test = test_dataset.targets.numpy().astype(np.int64)

    np.savez_compressed(
        DATA_PATH,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test
    )

    print("\nDataset prepared successfully!")
    print(f"Saved to: {DATA_PATH}")
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_test shape: {y_test.shape}")
    print(f"Seed used: {SEED}")


if __name__ == "__main__":
    main()
