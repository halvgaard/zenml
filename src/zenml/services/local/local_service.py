#  Copyright (c) ZenML GmbH 2022. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at:
#
#       https://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
#  or implied. See the License for the specific language governing
#  permissions and limitations under the License.

from abc import abstractmethod

import os
import pathlib
import psutil
import requests  # type: ignore

import socket
import subprocess

import tempfile
from typing import Any, Dict, List, Optional, Tuple

import signal
import sys

from zenml.services.base_service import (
    ServiceState,
    ServiceConfig,
    ServiceStatus,
    ServiceEndpointProtocol,
    ServiceEndpointConfig,
    ServiceEndpointStatus,
    BaseServiceEndpoint,
    BaseService,
    ServiceEndpointHealthMonitorConfig,
    BaseServiceEndpointHealthMonitor,
)


from zenml.utils.enum_utils import StrEnum
from zenml.logger import LOG_FILE, get_logger

logger = get_logger(__name__)

SERVICE_PORT_RANGE = (8000, 65535)
DEFAULT_HTTP_HEALTHCHECK_TIMEOUT = 5


class HttpEndpointHealthMonitorConfig(ServiceEndpointHealthMonitorConfig):
    """HTTP service endpoint health monitor configuration.

    Attributes:
        healthcheck_uri_path: URI subpath to use to perform service endpoint
            healthchecks. If not set, the service endpoint URI will be used
            instead.
        http_status_code: HTTP status code to expect in the health check
            response.
        http_timeout: HTTP health check request timeout in seconds.
    """

    healthcheck_uri_path: Optional[str]
    http_status_code: Optional[int] = 200
    http_timeout: Optional[int] = DEFAULT_HTTP_HEALTHCHECK_TIMEOUT


class HttpEndpointHealthMonitor(BaseServiceEndpointHealthMonitor):
    """HTTP service endpoint health monitor."""

    CONFIG_TYPE = HttpEndpointHealthMonitorConfig

    def __init__(
        self,
        config: HttpEndpointHealthMonitorConfig,
    ) -> None:
        super().__init__(config)

    def get_healthcheck_uri(
        self, endpoint: "BaseServiceEndpoint"
    ) -> Optional[str]:
        uri = endpoint.status.uri
        if not uri:
            return None
        return f"{uri}{self.config.healthcheck_uri_path or '/'}"

    def check_endpoint_status(
        self, endpoint: "BaseServiceEndpoint"
    ) -> Tuple[ServiceState, Optional[str]]:
        """Run a HTTP endpoint API healthcheck

        Returns:
            The operational state of the external HTTP endpoint and an
            optional error message, if an error is encountered while checking
            the HTTP endpoint status.
        """
        check_uri = self.get_healthcheck_uri(endpoint)
        if not check_uri:
            return ServiceState.ERROR, "no HTTP healthcheck URI available"

        logger.debug("Running HTTP healthcheck for URI: %s", check_uri)

        error = None

        try:
            r = requests.head(
                check_uri,
                timeout=self.config.http_timeout,
            )
            if r.status_code == self.config.http_status_code:
                # the endpoint is healthy
                return ServiceState.ACTIVE, None
            error = f"HTTP endpoint healthcheck returned unexpected status code: {r.status_code}"
        except requests.ConnectionError as e:
            error = f"HTTP endpoint healthcheck connection error: {str(e)}"
        except requests.Timeout as e:
            error = f"HTTP endpoint healthcheck request timed out: {str(e)}"
        except requests.RequestException as e:
            error = (
                f"unexpected error encountered while running HTTP endpoint "
                f"healthcheck: {str(e)}"
            )

        return ServiceState.ERROR, error


class TCPEndpointHealthMonitorConfig(ServiceEndpointHealthMonitorConfig):
    """TCP service endpoint health monitor configuration."""

    ...


class TCPEndpointHealthMonitor(BaseServiceEndpointHealthMonitor):
    """TCP service endpoint health monitor."""

    CONFIG_TYPE = TCPEndpointHealthMonitorConfig

    def __init__(
        self,
        config: TCPEndpointHealthMonitorConfig,
    ) -> None:
        super().__init__(config)

    @classmethod
    def port_is_open(cls, hostname: str, port: int) -> bool:
        """Check if a TCP port is open on a remote host.

        Args:
            hostname: hostname of the remote machine
            port: TCP port number

        Returns:
            True if the port is open, False otherwise
        """
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            result = sock.connect_ex((hostname, port))
            return result == 0

    def check_endpoint_status(
        self, endpoint: "BaseServiceEndpoint"
    ) -> Tuple[ServiceState, Optional[str]]:
        """Run a TCP endpoint healthcheck

        Returns:
            The operational state of the external TCP endpoint and an
            optional error message, if an error is encountered while checking
            the TCP endpoint status.
        """
        if not endpoint.status.port:
            return ServiceState.ERROR, "no TCP port information available"

        logger.debug(
            "Running TCP healthcheck for TCP port: %d", endpoint.status.port
        )

        if self.port_is_open(endpoint.status.hostname, endpoint.status.port):
            # the endpoint is healthy
            return ServiceState.ACTIVE, None

        return (
            ServiceState.ERROR,
            "TCP endpoint healthcheck error: TCP port is not "
            "open or not accessible",
        )


class LocalDaemonServiceEndpointConfig(ServiceEndpointConfig):
    """Local daemon service endpoint configuration.

    Attributes:
        protocol: the TCP protocol implemented by the service endpoint
        port: preferred TCP port value for the service endpoint. If the port
            is in use when the service is started, setting `allocate_port` to
            True will also try to allocate a new port value, otherwise an
            exception will be raised.
        allocate_port: set to True to allocate a free TCP port for the
            service endpoint automatically.
    """

    protocol: Optional[ServiceEndpointProtocol] = ServiceEndpointProtocol.TCP
    port: Optional[int]
    allocate_port: Optional[bool] = True


class LocalDaemonServiceEndpointStatus(ServiceEndpointStatus):
    """Local daemon service endpoint status.

    Attributes:
    """

    ...


class LocalDaemonServiceEndpoint(BaseServiceEndpoint):
    """A service endpoint exposed by a local daemon process.

    This class extends the base service endpoint class with functionality
    concerning the life-cycle management and tracking of endpoints exposed
    by external services implemented as local daemon processes.
    """

    STATUS_TYPE = LocalDaemonServiceEndpointStatus
    CONFIG_TYPE = LocalDaemonServiceEndpointConfig
    # TODO: allow both TCP and HTTP monitors
    MONITOR_TYPE = HttpEndpointHealthMonitor

    def __init__(
        self,
        config: LocalDaemonServiceEndpointConfig,
        monitor: Optional[HttpEndpointHealthMonitor] = None,
    ) -> None:
        super().__init__(config, monitor)

    @classmethod
    def port_is_available(cls, port: int) -> bool:
        """Check if a TCP port is available on the local machine.

        Args:
            port: TCP port number

        Returns:
            True if the port is available, False otherwise
        """
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind(("", port))
                sock.close()
                return True
            except OSError:
                return False

    def _lookup_free_port(self) -> int:
        """Search for a free TCP port for the service endpoint.

        If a preferred TCP port value is explicitly requested through the
        endpoint configuration, it will be checked first. If a port was
        previously used the last time the service was running (i.e. as
        indicated in the service endpoint status), it will be checked next for
        availability.

        As a last resort, this call will search for a free TCP port, if
        `allocate_port` is set to True in the endpoint configuration.

        Returns:
            An available TCP port number

        Raises:
            IOError: if the preferred TCP port is busy and `allocate_port` is
            disabled in the endpoint configuration, or if no free TCP port
            could be otherwise allocated.
        """

        # If a port value is explicitly configured, attempt to use it first
        if self.config.port:
            if self.port_is_available(self.config.port):
                return self.config.port
            if not self.config.allocate_port:
                raise IOError(f"TCP port {self.config.port} is not available.")

        # Attempt to reuse the port used when the services was last running
        if self.status.port and self.port_is_available(self.status.port):
            return self.status.port

        # As a last resort, try to find a free port in the range
        for port in range(*SERVICE_PORT_RANGE):
            if self.port_is_available(port):
                return port
        raise IOError(
            "No free TCP ports found in the range %d - %d",
            SERVICE_PORT_RANGE[0],
            SERVICE_PORT_RANGE[1],
        )

    def prepare_for_start(self) -> None:
        """Prepare the service endpoint for starting.

        This method is called before the service is started.
        """
        self.status.protocol = self.config.protocol
        self.status.hostname = "localhost"
        self.status.port = self._lookup_free_port()


class LocalDaemonServiceConfig(ServiceConfig):
    """Local daemon service configuration.

    Attributes:
        silent_daemon: set to True to suppress the output of the daemon
            (i.e. redirect stdout and stderr to /dev/null). If False, the
            daemon output will be redirected to a logfile.
        graceful_shutdown_signal: the signal to send to the daemon process
            to shut it down gracefully.
        forceful_shutdown_signal: the signal to send to the daemon process
            to shut it down forcefully.
        shutdown_signal_session: set to True to signal the entire process group
            of the daemon process, instead of only the daemon process itself,
            when shutting it down.
    """

    silent_daemon: Optional[bool] = False
    graceful_shutdown_signal: Optional[signal.Signals] = signal.SIGTERM
    forceful_shutdown_signal: Optional[signal.Signals] = signal.SIGKILL
    shutdown_signal_session: Optional[bool] = False


SERVICE_DAEMON_CONFIG_FILE_NAME = "service.json"
SERVICE_DAEMON_LOG_FILE_NAME = "service.log"


class LocalDaemonServiceStatus(ServiceStatus):
    """Local daemon service status.

    Attributes:
        pid: the current process ID of the service daemon
        runtime_path: the path where the service daemon runtime files (the
            configuration file used to start the service daemon and the
            logfile) are located
        silent_daemon: flag indicating whether the output of the daemon
            is suppressed.
    """

    pid: Optional[int]
    runtime_path: Optional[str]
    silent_daemon: Optional[bool]

    @property
    def config_file(self) -> Optional[str]:
        """Get the path to the configuration file used to start the service
        daemon.

        Returns:
            The path to the configuration file, or an empty string, if the
            service has never been started before.
        """
        if not self.runtime_path:
            return ""
        return os.path.join(self.runtime_path, SERVICE_DAEMON_CONFIG_FILE_NAME)

    @property
    def log_file(self) -> Optional[str]:
        """Get the path to the log file where the service output is/has been
        logged.

        Returns:
            The path to the log file, or an empty string, if the
            service has never been started before, or if the service daemon
            output is suppressed.
        """
        if not self.runtime_path or self.silent_daemon:
            return ""
        return os.path.join(self.runtime_path, SERVICE_DAEMON_LOG_FILE_NAME)


class LocalDaemonService(BaseService):
    """A service represented by a local daemon process.

    This class extends the base service class with functionality concerning
    the life-cycle management and tracking of external services implemented as
    local daemon processes.

    The default implementation is to launch a python wrapper that
    instantiates the same LocalDaemonService object from the
    serialized configuration and calls its `run` method.
    """

    CONFIG_TYPE = LocalDaemonServiceConfig
    STATUS_TYPE = LocalDaemonServiceStatus
    ENDPOINT_TYPE = LocalDaemonServiceEndpoint

    def __init__(
        self,
        config: LocalDaemonServiceConfig,
        endpoint: Optional[LocalDaemonServiceEndpoint] = None,
    ) -> None:
        super().__init__(config, endpoint)

    def check_status(self) -> Tuple[ServiceState, Optional[str]]:
        """Check the the current operational state of the daemon process.

        Returns:
            The operational state of the daemon process and an optional error
            message, if an error is encountered while checking its status.
        """

        if not self.status.pid or not psutil.pid_exists(self.status.pid):
            self.status.pid = None
            return ServiceState.INACTIVE, "service daemon is not running"

        # the daemon is running
        return ServiceState.ACTIVE, None

    def _get_daemon_cmd(self) -> Tuple[Tuple[str], Dict[str, str]]:
        """Get the command to run to launch the service daemon.

        The default implementation provided by this class is the following:

          * the configuration describing this LocalDaemonService instance
          is serialized as JSON and saved to a temporary file
          * the python executable wrapper script is
          the configuration is to launch a
        python wrapper that instantiates the same LocalDaemonService object

        Subclasses that need a different command to launch the service daemon
        should overrride this method.

        Returns:
            Command needed to launch the daemon process and the environment
            variables to set for it, in the formats accepted by
            subprocess.Popen.
        """
        # to avoid circular imports, import here
        import zenml.services.local.local_daemon_entrypoint as daemon_entrypoint

        self.status.silent_daemon = self.config.silent_daemon
        # reuse the config file and logfile location from a previous run,
        # if available
        if not self.status.runtime_path or not os.path.exists(
            self.status.runtime_path
        ):
            self.status.runtime_path = tempfile.mkdtemp(prefix="zenml-service-")
        with open(self.status.config_file, "w") as f:
            f.write(self.to_json())

        command = (
            sys.executable,
            "-m",
            daemon_entrypoint.__name__,
            "--config-file",
            self.status.config_file,
        )
        if self.status.log_file:
            pathlib.Path(self.status.log_file).touch()
            command += ("--log-file", self.status.log_file)

        command_env = os.environ.copy()
        # command_env[_SERVER_MODEL_PATH] = local_uri

        return command, command_env

    def _start_daemon(self) -> None:
        """Start the service daemon process associated with this service."""

        logger.info("Starting daemon for service '%s'...", self.config.name)

        if self.status.pid and psutil.pid_exists(self.status.pid):
            # service daemon is already running
            logger.debug(
                "Daemon process for service '%s' is already running with PID %d",
                self.config.name,
                self.status.pid,
            )
            return

        if self.endpoint:
            self.endpoint.prepare_for_start()

        command, command_env = self._get_daemon_cmd()
        logger.debug(
            "Running command to start daemon for service '%s': %s",
            self.config.name,
            " ".join(command),
        )
        p = subprocess.Popen(command, env=command_env)
        self.status.pid = p.pid
        logger.debug(
            "Daemon process for service '%s' started with PID: %d",
            self.config.name,
            self.status.pid,
        )

    def _stop_daemon(self, force: Optional[bool] = False) -> None:
        """Stop the service daemon process associated with this service.

        Args:
            force: if True, the service daemon will be forcefully stopped
        """
        logger.info("Stopping daemon for service '%s' ...", self.config.name)

        if not self.status.pid or not psutil.pid_exists(self.status.pid):
            # service daemon is not running
            logger.debug(
                "Daemon process for service '%s' no longer running",
                self.config.name,
            )
            self.status.pid = None
            return

        s = self.config.graceful_shutdown_signal
        if force:
            s = self.config.forceful_shutdown_signal
        logger.debug(
            "Sending signal %s to daemon process for service '%s' to stop",
            s.name,
            self.config.name,
        )

        if self.config.shutdown_signal_session:
            pgrp = os.getpgid(self.status.pid)
            os.killpg(pgrp, s)
        else:
            os.kill(self.status.pid, s)

    def provision(self) -> None:
        self._start_daemon()

    def deprovision(self, force: Optional[bool] = False) -> None:
        self._stop_daemon(force)

    @abstractmethod
    def run(self) -> None:
        """Run the service daemon process associated with this service.

        Subclasses must implement this method to provide the service daemon
        functionality.
        """
