#  Copyright (c) ZenML GmbH 2022. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at:
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
#  or implied. See the License for the specific language governing
#  permissions and limitations under the License.
import click
import numpy as np

from zenml.environment import Environment
from zenml.integrations.mlflow.mlflow_environment import MLFLOW_ENVIRONMENT_NAME

from pipeline import (
    TrainerConfig,
    mlflow_example_pipeline,
    importer_mnist,
    normalizer,
    tf_trainer,
    tf_evaluator,
    predictor,
    batch_inference,
    inference_evaluator,
)
from zenml.repository import Repository
from zenml.services import BaseService


def get_service(step_name: str) -> BaseService:
    """Load a service artifact saved in the repository during the last execution
    of the pipeline step with the given step name.

    Args:
        step_name: pipeline step name

    Returns:
        BaseService: service artifact
    """
    repo = Repository()
    pipe = repo.get_pipelines()[-1]
    step = pipe.runs[-1].get_step(name=step_name)
    for artifact_name, artifact_view in step.outputs.items():
        # filter out anything but service artifacts
        if artifact_view.type == "ServiceArtifact":
            return artifact_view.read()


@click.command()
@click.option("--epochs", default=5, help="Number of epochs for training")
@click.option("--lr", default=0.003, help="Learning rate for training")
@click.option(
    "--stop-service",
    is_flag=True,
    default=False,
    help="Stop the prediction service when done",
)
def main(epochs: int, lr: float, stop_service: bool):
    """Run the MLflow example pipeline"""

    if stop_service:
        service = get_service(step_name="predictor")
        if service:
            service.stop(timeout=10)
        return

    # Initialize a pipeline run
    run = mlflow_example_pipeline(
        importer=importer_mnist(),
        normalizer=normalizer(),
        trainer=tf_trainer(config=TrainerConfig(epochs=epochs, lr=lr)),
        evaluator=tf_evaluator(),
        predictor=predictor(),
        batch_inference=batch_inference(),
        inference_evaluator=inference_evaluator(),
    )

    run.run()

    mlflow_env = Environment()[MLFLOW_ENVIRONMENT_NAME]
    print(
        "You can run:\n "
        f"    mlflow ui --backend-store-uri {mlflow_env.tracking_uri}\n"
        "To inspect your experiment runs within the mlflow ui.\n"
        "You can find your runs tracked within the `mlflow_example_pipeline`"
        "experiment. Here you'll also be able to compare two or more runs.\n\n"
    )

    service = get_service(step_name="predictor")

    if not service.is_running:
        service.start(timeout=10)
    print("Sending inference request to MLflow deployment server...")
    print(f"Result is\n: {service.predict(np.random.rand(1, 28, 28))}\n\n", )
    print(
        f"The MLflow prediction server is running locally as a daemon process "
        f"and accepts inference requests at: {service.prediction_uri}. "
        f"To stop the service, re-run the same command and supply the "
        f"`--stop-service` argument."
    )


if __name__ == "__main__":
    main()
