import os
import random
import numpy as np
from sklearn.datasets import fetch_openml

SEED = 42
DATA_PATH = "data/mnist.npz"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def main():
    set_seed(SEED)

    os.makedirs("data", exist_ok=True)

    print("Downloading/loading MNIST...")

    # Load MNIST from OpenML using scikit-learn.
    # The resulting data is converted to the same format expected by train.py.
    mnist = fetch_openml(
        "mnist_784",
        version=1,
        as_frame=False,
        parser="auto"
    )

    X = mnist.data.astype(np.float32) / 255.0
    y = mnist.target.astype(np.int64)

    # MNIST contains 70,000 samples:
    # first 60,000 are used for training and last 10,000 for testing.
    X_train = X[:60000]
    y_train = y[:60000]

    X_test = X[60000:]
    y_test = y[60000:]

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
