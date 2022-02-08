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

import os
from typing import Any, Type

from zenml.integrations.mlflow.mlflow_predictor import MLFlowPredictionService
from zenml.artifacts import ServiceArtifact
from zenml.io import fileio
from zenml.materializers.base_materializer import BaseMaterializer

SERVICE_CONFIG_FILENAME = "service.json"


class MLFlowPredictorMaterializer(BaseMaterializer):
    """Materializer to read/write MLflow prediction services."""

    ASSOCIATED_TYPES = (MLFlowPredictionService,)
    ASSOCIATED_ARTIFACT_TYPES = (ServiceArtifact,)

    def handle_input(self, data_type: Type[Any]) -> MLFlowPredictionService:
        """Creates and returns an MLflow prediction service instantiated from
        the service configuration saved as artifact.

        Returns:
            An MLflow prediction service.
        """
        super().handle_input(data_type)
        filepath = os.path.join(self.artifact.uri, SERVICE_CONFIG_FILENAME)
        with fileio.open(filepath, "rb") as f:
            service = MLFlowPredictionService.from_json(f.read())
        return service

    def handle_return(self, service: MLFlowPredictionService) -> None:
        """Writes an MLflow prediction service.

        The configuration of the input MLflow prediction service instance is
        serialized and saved as an artifact. The configuration can be loaded
        later on and used to create a new MLflow prediction service instance
        that is equivalent with the input one.

        Args:
            service: An MLflow prediction service instance.
        """
        super().handle_return(service)
        filepath = os.path.join(self.artifact.uri, SERVICE_CONFIG_FILENAME)
        with fileio.open(filepath, "wb") as f:
            f.write(service.to_json())
