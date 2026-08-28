import os

import mlflow


TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://127.0.0.1:5000"
)

MODEL_NAME = "MNIST_Reproducibility_Model"

client = mlflow.tracking.MlflowClient()

def main():

    mlflow.set_tracking_uri(
        TRACKING_URI
    )

    if not os.path.exists(
        "artifacts/latest_run_id.txt"
    ):
        raise FileNotFoundError(
            "latest_run_id.txt not found. "
            "Run train.py first."
        )

    with open(
        "artifacts/latest_run_id.txt",
        "r"
    ) as file:

        run_id = file.read().strip()

    model_uri = (
        f"runs:/{run_id}/model"
    )

    print(
        f"Registering model from run: {run_id}"
    )

    registered_model = (
        mlflow.register_model(
            model_uri=model_uri,
            name=MODEL_NAME
        )
    )

    print(
        f"Model registered successfully!"
    )

    print(
        f"Model name: {MODEL_NAME}"
    )

    print(
        f"Model version: "
        f"{registered_model.version}"
    )
      
    try:
        client.transition_model_version_stage(
            name=MODEL_NAME,
            version=registered_model.version,
            stage="Staging"
        )
        print(f"Model version {registered_model.version} transitioned to Staging.")

    except AttributeError:
        client.set_registered_model_alias(
            name=MODEL_NAME,
            alias="staging",
            version=registered_model.version
        )
        print(f"Stages unsupported in this MLflow version — set alias 'staging' on version {registered_model.version} instead.")


if __name__ == "__main__":

    main()