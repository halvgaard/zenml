from typing import Any, Dict, Optional, Union

import numpy as np
import requests  # type: ignore
from mlflow.pyfunc.backend import PyFuncBackend  # type: ignore

from zenml.logger import get_logger
from zenml.services import (
    HTTPEndpointHealthMonitor,
    HTTPEndpointHealthMonitorConfig,
    LocalDaemonService,
    LocalDaemonServiceConfig,
    LocalDaemonServiceEndpoint,
    LocalDaemonServiceEndpointConfig,
    ServiceEndpointProtocol,
    ServiceType,
)

logger = get_logger(__name__)

MLFLOW_PREDICTION_URL_PATH = "/invocations"
MLFLOW_HEALTHCHECK_URL_PATH = "/ping"

MLSERVER_PREDICTION_URL_PATH = "/inference"
MLSERVER_HEALTHCHECK_URL_PATH = "/healtz"


class MLFlowDeploymentEndpointConfig(LocalDaemonServiceEndpointConfig):
    """MLflow daemon service endpoint configuration.

    Attributes:
        prediction_uri_path: URI subpath for prediction requests
    """

    prediction_uri_path: Optional[str]


class MLFlowDeploymentEndpoint(LocalDaemonServiceEndpoint):
    """A service endpoint exposed by the MLflow deployment daemon."""

    config: MLFlowDeploymentEndpointConfig
    monitor: HTTPEndpointHealthMonitor

    @property
    def prediction_uri(self) -> Optional[str]:
        uri = self.status.uri
        if not uri:
            return None
        return f"{uri}{self.config.prediction_uri_path or '/'}"


class MLFlowDeploymentConfig(LocalDaemonServiceConfig):
    """MLflow model deployment configuration.

    Attributes:
        model_uri: URI of the MLflow model to serve
        workers: number of workers to use for the prediction service
        mlserver: set to True to use the MLflow MLServer backend (see
            https://github.com/SeldonIO/MLServer). If False, the
            MLflow builtin scoring server will be used.
    """

    model_uri: str
    workers: Optional[int] = 1
    mlserver: Optional[bool] = False


class MLFlowDeploymentService(LocalDaemonService):

    SERVICE_TYPE = ServiceType(
        name="mlflow-deployment",
        type="model-serving",
        flavor="mlflow",
        description="MLflow prediction service",
    )

    config: MLFlowDeploymentConfig
    endpoint: MLFlowDeploymentEndpoint

    def __init__(
        self,
        config: Union[MLFlowDeploymentConfig, Dict[str, Any]],
        **attrs: Any,
    ) -> None:
        # ensure that the endpoint is created before the service is initialized
        # TODO [HIGH]: implement a service factory or builder for MLflow
        #   deployment services
        if (
            isinstance(config, MLFlowDeploymentConfig)
            and "endpoint" not in attrs
        ):
            if config.mlserver:
                endpoint = MLFlowDeploymentEndpoint(
                    config=MLFlowDeploymentEndpointConfig(
                        protocol=ServiceEndpointProtocol.HTTP,
                        prediction_uri_path=MLSERVER_PREDICTION_URL_PATH,
                    ),
                    monitor=HTTPEndpointHealthMonitor(
                        config=HTTPEndpointHealthMonitorConfig(
                            healthcheck_uri_path=MLSERVER_HEALTHCHECK_URL_PATH
                        )
                    ),
                )
            else:
                endpoint = MLFlowDeploymentEndpoint(
                    config=MLFlowDeploymentEndpointConfig(
                        protocol=ServiceEndpointProtocol.HTTP,
                        prediction_uri_path=MLFLOW_PREDICTION_URL_PATH,
                    ),
                    monitor=HTTPEndpointHealthMonitor(
                        config=HTTPEndpointHealthMonitorConfig(
                            healthcheck_uri_path=MLFLOW_HEALTHCHECK_URL_PATH
                        )
                    ),
                )
            attrs["endpoint"] = endpoint
        super().__init__(config=config, **attrs)

    def run(self) -> None:
        logger.info(
            "Starting MLflow prediction service as blocking "
            "process... press CTRL+C once to stop it."
        )

        self.endpoint.prepare_for_start()

        try:
            backend = PyFuncBackend(
                config={},
                no_conda=True,
                workers=self.config.workers,
                install_mlflow=False,
            )
            backend.serve(
                model_uri=self.config.model_uri,
                port=self.endpoint.status.port,
                host="localhost",
                enable_mlserver=self.config.mlserver,
            )
        except KeyboardInterrupt:
            logger.info(
                "MLflow prediction service stopped. Resuming normal execution."
            )

    def predict(self, request: np.ndarray) -> np.ndarray:
        """Make a prediction using the service.

        Args:
            request: a numpy array representing the request

        Returns:
            A numpy array representing the prediction returned by the service.
        """
        if not self.is_running:
            raise Exception(
                "MLflow prediction service is not running. "
                "Please start the service before making predictions."
            )

        response = requests.post(
            self.endpoint.prediction_uri,
            json={"instances": request.tolist()},
        )
        response.raise_for_status()
        return np.array(response.json())
