import os

import mlflow


TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://127.0.0.1:5000"
)

MODEL_NAME = "MNIST_Reproducibility_Model"

# FIX 1 -------------------------------------------------------------------
# The MlflowClient was previously constructed HERE, at module import time,
# before set_tracking_uri() runs inside main(). MlflowClient captures the
# tracking URI at construction, so it silently pointed at the default
# localhost:5000 rather than TRACKING_URI. Any transition or alias call then
# went to the wrong server (or failed outright).
#
# The client is now created inside main(), after the URI is set.
# -------------------------------------------------------------------------


def main():

    mlflow.set_tracking_uri(
        TRACKING_URI
    )

    print(f"MLflow URI: {TRACKING_URI}")

    # created AFTER set_tracking_uri, so it inherits the correct URI
    client = mlflow.tracking.MlflowClient()

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

    print("Model registered successfully!")
    print(f"Model name: {MODEL_NAME}")
    print(f"Model version: {registered_model.version}")

    # FIX 2 -----------------------------------------------------------------
    # transition_model_version_stage() is REMOVED in MLflow 3.x, so calling it
    # raises MlflowException or AttributeError depending on version. The old
    # code caught AttributeError only, so on MLflow 3.x the script crashed
    # instead of falling back to aliases.
    #
    # Catching Exception covers both, and the message states which path ran.
    # -----------------------------------------------------------------------
    try:
        client.transition_model_version_stage(
            name=MODEL_NAME,
            version=registered_model.version,
            stage="Staging"
        )
        print(
            f"Model version {registered_model.version} "
            f"transitioned to Staging."
        )

    except Exception as error:
        print(
            f"Stages unavailable in this MLflow version ({type(error).__name__}). "
            f"Falling back to the aliases API."
        )

        client.set_registered_model_alias(
            name=MODEL_NAME,
            alias="staging",
            version=registered_model.version
        )

        client.set_model_version_tag(
            name=MODEL_NAME,
            version=registered_model.version,
            key="stage",
            value="Staging"
        )

        print(
            f"Set alias 'staging' on version "
            f"{registered_model.version}."
        )


if __name__ == "__main__":

    main()
