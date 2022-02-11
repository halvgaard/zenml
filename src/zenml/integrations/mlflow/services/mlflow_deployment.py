from mlflow.pyfunc.backend import PyFuncBackend  # type: ignore

import numpy as np
from pydantic import BaseModel
import requests  # type: ignore
from typing import Optional
from zenml.logger import get_logger
from zenml.services import (
    LocalDaemonServiceConfig,
    LocalDaemonService,
    LocalDaemonServiceEndpointConfig,
    LocalDaemonServiceEndpoint,
    ServiceEndpointProtocol,
    HttpEndpointHealthMonitorConfig,
    HttpEndpointHealthMonitor,
    ServiceType,
)


logger = get_logger(__name__)

MLFLOW_PREDICTION_URL_PATH = "/invocations"
MLFLOW_HEALTHCHECK_URL_PATH = "/ping"


class MLFlowDeploymentEndpointConfig(LocalDaemonServiceEndpointConfig):
    """MLflow daemon service endpoint configuration.

    Attributes:
        prediction_uri_path: URI subpath for prediction requests
    """

    name: str = "MLFlow Deployment Endpoint"
    protocol: ServiceEndpointProtocol = ServiceEndpointProtocol.HTTP
    prediction_uri_path: Optional[str]


class MLFlowDeploymentEndpoint(LocalDaemonServiceEndpoint):
    """A service endpoint exposed by the MLflow deployment daemon."""

    CONFIG_TYPE = MLFlowDeploymentEndpointConfig
    MONITOR_TYPE = HttpEndpointHealthMonitor

    def __init__(
        self,
        config: MLFlowDeploymentEndpointConfig,
        monitor: Optional[HttpEndpointHealthMonitor] = None,
    ) -> None:
        if not monitor:
            monitor = HttpEndpointHealthMonitor(
                HttpEndpointHealthMonitorConfig(
                    healthcheck_uri_path=MLFLOW_HEALTHCHECK_URL_PATH
                )
            )
        super().__init__(config, monitor)

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
        mlserver: set to True to use the MLflow MLServer backend. If False, the
            MLflow builtin scoring server will be used.
    """

    model_uri: str
    workers: Optional[int] = 1
    mlserver: Optional[
        bool
    ] = False  # see: https://github.com/SeldonIO/MLServer


class MLFlowDeploymentService(LocalDaemonService):

    SERVICE_TYPE = ServiceType(
        name="mlflow-deployment",
        type="model-serving",
        flavor="mlflow",
        description="MLflow prediction service",
    )
    CONFIG_TYPE = MLFlowDeploymentConfig
    ENDPOINT_TYPE = MLFlowDeploymentEndpoint

    def __init__(
        self,
        config: MLFlowDeploymentConfig,
        endpoint: Optional[LocalDaemonServiceEndpoint] = None,
    ) -> None:
        if endpoint is None:
            endpoint = MLFlowDeploymentEndpoint(
                MLFlowDeploymentEndpointConfig(
                    prediction_uri_path=MLFLOW_PREDICTION_URL_PATH
                )
            )
        super().__init__(config, endpoint)

    @classmethod
    def type(cls) -> ServiceType:
        """The MLflow local prediction service type."""
        return cls.SERVICE_TYPE

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
            pass

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
        logger.debug(response.json())
        return np.array(response.json())
