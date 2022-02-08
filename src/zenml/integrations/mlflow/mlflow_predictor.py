import click

from mlflow.pyfunc.backend import PyFuncBackend

from mlflow.models import Model
from mlflow.models.flavor_backend_registry import get_flavor_backend
from mlflow import pyfunc

import numpy as np

import os
import psutil
from pydantic import BaseModel
import requests

import socket
import subprocess

import tempfile
import time
from typing import Dict, List, Optional, Tuple

import signal
import sys
from zenml.utils.enum_utils import StrEnum
from zenml.logger import get_logger

logger = get_logger(__name__)


class ServiceState(StrEnum):
    """All possible types a `ServiceState` can have."""

    ACTIVE = "active"
    INACTIVE = "inactive"
    ERROR = "error"


class MLFlowPredictionServiceConfig(BaseModel):
    """[summary]

    Attributes:
        model_uri: [description]
        workers: [description]
        mlserver: [description]
        port: preferred TCP port value for the prediction service. If the port
            is in use when the service is started, setting `allocate_port` to
            True will also try to allocate a new port value, otherwise an
            exception will be raised.
        allocate_port: set to True to allocate a free TCP port for the
            prediction service.
    """

    model_uri: str
    workers: Optional[int] = 1
    mlserver: Optional[
        bool
    ] = False  # see: https://github.com/SeldonIO/MLServer
    port: Optional[int] = None
    allocate_port: Optional[bool] = True


class MLFlowPredictionServiceStatus:
    def __init__(self) -> None:
        self.runtime_state = ServiceState.INACTIVE
        self.service_state = ServiceState.INACTIVE
        self.prediction_uri = None
        self.healthcheck_uri = None
        self.pid = None
        self.port = None
        self.last_error = None


MLFLOW_PORT_RANGE = (8000, 65535)
MLFLOW_PREDICTION_URL_PATH = "/invocations"
MLFLOW_HEALTHCHECK_URL_PATH = "/ping"


class MLFlowPredictionService:
    def __init__(self, config: MLFlowPredictionServiceConfig) -> None:
        self.config = config
        self.status = MLFlowPredictionServiceStatus()
        self.admin_state = ServiceState.INACTIVE

    @classmethod
    def port_is_available(cls, port: int) -> bool:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.bind(("", port))
            sock.close()
            return True
        except OSError:
            return False

    def allocate_port(self) -> int:
        """Allocate a free TCP port for the prediction service."""

        # If a port value is explicitly configured, attempt to use it first
        if self.config.port:
            if self.port_is_available(self.config.port):
                return self.config.port
            if not self.allocate_port:
                raise IOError(f"TCP port {self.config.port} is not available.")

        # Attempt to reuse the port used when the services was last running
        if self.status.port and self.port_is_available(self.status.port):
            return self.status.port

        # As a last resort, try to find a free port in the range
        for port in range(*MLFLOW_PORT_RANGE):
            if self.port_is_available(port):
                return port
        raise IOError(
            "No free TCP ports found in the range %d - %d",
            MLFLOW_PORT_RANGE[0],
            MLFLOW_PORT_RANGE[1],
        )

    def _get_cmd(self, config: str) -> Tuple[Tuple[str], Dict[str, str]]:

        command = (sys.executable, "-m", __name__, "--config", config)
        command_env = os.environ.copy()
        # command_env[_SERVER_MODEL_PATH] = local_uri

        return command, command_env

    def reconcile_service_status(self) -> None:

        # TODO: use admin state to determine what to do here

        if not self.status.pid:
            self.status.runtime_state = ServiceState.INACTIVE
            self.status.service_state = ServiceState.INACTIVE
            return

        if not psutil.pid_exists(self.status.pid):
            self.status.last_error = "prediction server process not running"
            logger.error(
                f"error waiting for MLflow prediction server to become active: {self.status.last_error}"
            )
            self.status.runtime_state = ServiceState.ERROR
            return

        try:
            r = requests.head(self.status.healthcheck_uri)
            if r.status_code == 200:
                self.status.last_error = None
                self.status.service_state = ServiceState.ACTIVE
                logger.info(
                    "Prediction service active and receiving requests at: %s",
                    self.status.prediction_uri,
                )
                return
            self.status.last_error = f"prediction service API returns HTTP status code {r.status_code}"
        except requests.ConnectionError as e:
            self.status.last_error = (
                f"cannot connect to prediction service API: {str(e)}"
            )

        self.status.service_state = ServiceState.ERROR

    def poll_service_status(self, timeout: int) -> None:
        if not self.status.pid:
            return

        while timeout > 0:
            self.reconcile_service_status()
            if (
                self.status.runtime_state == ServiceState.ACTIVE
                and self.status.service_state == ServiceState.ACTIVE
            ):
                return
            time.sleep(1)
            timeout -= 1
        logger.error(
            f"Timed out waiting for MLflow prediction service to become active: {self.status.last_error}"
        )

    def to_json(self) -> str:
        """Serialize the service configuration in JSON format.

        Returns:
            str: serialized service configuration
        """
        # override the configuration to include the last allocated port, if any
        config = self.config.copy(deep=True)
        if not config.port and config.allocate_port and self.status.port:
            config.port = self.status.port
        return config.json(indent=4)

    @classmethod
    def from_json(cls, json_str: str) -> "MLFlowPredictionService":
        """Instantiate a service instance from a serialized service
        configuration.

        Args:
            json_str: serialized service configuration

        Returns:
            A service instance created from the serialized configuration.
        """
        config = MLFlowPredictionServiceConfig.from_json(json_str)
        return cls(config)

    def start_daemon(self, wait_active_timeout: Optional[int] = 0) -> None:
        if (
            self.status.runtime_state == ServiceState.ACTIVE
            and self.status.service_state == ServiceState.ACTIVE
        ):
            logger.debug("MLflow prediction service already running")
            return

        logger.info("Starting MLflow prediction service as daemon...")

        port = self.allocate_port()

        # override the configuration to enforce the allocated port
        config = self.config.copy(deep=True)
        config.port = port
        config.allocate_port = False

        with tempfile.NamedTemporaryFile(
            mode="w", delete=False, suffix=".json"
        ) as f:
            f.write(config.json(indent=4))
            cfg_file = f.name

        command, command_env = self._get_cmd(cfg_file)
        logger.debug("Running service daemon command: %s", command)
        p = subprocess.Popen(command, env=command_env, start_new_session=True)
        self.status.runtime_state = ServiceState.ACTIVE
        self.status.pid = p.pid
        self.status.port = port
        self.status.prediction_uri = (
            f"http://localhost:{port}{MLFLOW_PREDICTION_URL_PATH}"
        )
        self.status.healthcheck_uri = (
            f"http://localhost:{port}{MLFLOW_HEALTHCHECK_URL_PATH}"
        )
        logger.debug(
            "Prediction service started with PID: %d",
            self.status.pid,
        )
        if wait_active_timeout:
            logger.info(
                "Waiting for prediction service to become active...",
            )
            self.poll_service_status(wait_active_timeout)

    def start(self) -> None:
        if (
            self.status.runtime_state == ServiceState.ACTIVE
            and self.status.service_state == ServiceState.ACTIVE
        ):
            logger.debug("MLflow prediction service already running")
            return

        logger.info(
            "Starting MLflow prediction service as blocking "
            "process... press CTRL+C once to stop it."
        )

        port = self.allocate_port()

        self.status.port = port
        self.status.prediction_uri = f"http://localhost:{port}/invocations"
        self.status.runtime_state = ServiceState.ACTIVE
        self.status.service_state = ServiceState.ACTIVE

        try:
            backend = PyFuncBackend(
                config={},
                no_conda=True,
                workers=self.config.workers,
                install_mlflow=False,
            )
            backend.serve(
                model_uri=self.config.model_uri,
                port=port,
                host="localhost",
                enable_mlserver=self.config.mlserver,
            )
        except KeyboardInterrupt:
            logger.info(
                "MLflow prediction service stopped. Resuming normal execution."
            )
            pass

        self.status.runtime_state = ServiceState.INACTIVE
        self.status.service_state = ServiceState.INACTIVE

    def stop(self, wait_stopped_timeout: Optional[int] = 0) -> None:
        if (
            self.status.runtime_state == ServiceState.INACTIVE
            and self.status.service_state == ServiceState.INACTIVE
        ):
            logger.debug("MLflow prediction service no longer running")
            return

        if not self.status.pid:
            return
        pgrp = os.getpgid(self.status.pid)
        os.killpg(pgrp, signal.SIGINT)
        # os.kill(self.pid, signal.SIGTERM)
        # TODO: wait for the process to die
        self.status.runtime_state = ServiceState.INACTIVE
        self.status.service_state = ServiceState.INACTIVE
        self.status.last_error = None
        self.status.pid = None

    def is_running(self) -> bool:
        self.reconcile_service_status()
        return self.status.runtime_state == ServiceState.ACTIVE

    def is_active(self) -> bool:
        self.reconcile_service_status()
        return self.status.service_state == ServiceState.ACTIVE

    def same_as(self, other: "MLFlowPredictionService") -> bool:
        ...

    def equivalent_to(self, other: "MLFlowPredictionService") -> bool:
        ...

    def predict(self, request: np.ndarray) -> np.ndarray:
        """Make a prediction using the service.

        Args:
            request: a numpy array representing the request

        Returns:
            A numpy array representing the prediction returned by the service.
        """
        if not self.is_active():
            raise Exception(
                "MLflow prediction service is not running. "
                "Please start the service before making predictions."
            )

        response = requests.post(
            self.status.prediction_uri,
            json={"signature_name": self.config.signature_name, "instances": request},
        )
        response.raise_for_status()
        return response.json()["predictions"]


@click.command()
@click.option("--config", required=True)
def run(config: str) -> None:
    if os.path.isfile(config):
        logger.info("Loading service configuration from %s", config)
        # same code as materializer
        service_config = MLFlowPredictionServiceConfig.parse_file(config)
        logger.debug("Loaded service configuration: %s", service_config)
        MLFlowPredictionService(service_config).start()


if __name__ == "__main__":
    # os.setpgrp()
    # os.setsid()
    run()
