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
import numpy as np

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


if __name__ == "__main__":

    # Initialize a pipeline run
    run = mlflow_example_pipeline(
        importer=importer_mnist(),
        normalizer=normalizer(),
        trainer=tf_trainer(config=TrainerConfig(epochs=5, lr=0.0003)),
        evaluator=tf_evaluator(),
        predictor=predictor(),
        batch_inference=batch_inference(),
        inference_evaluator=inference_evaluator(),
    )

    run.run()

    service = get_service(step_name="predictor")

    if not service.is_running:
        service.start(timeout=10)
    print(service.predict(np.random.rand(1, 28, 28)))
