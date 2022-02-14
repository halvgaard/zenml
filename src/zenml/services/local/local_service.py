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

import os
import pathlib
import signal
import subprocess
import sys
import tempfile
from abc import abstractmethod
from typing import Dict, Optional, Tuple

import psutil

from zenml.logger import get_logger
from zenml.services.local.local_service_endpoint import (
    LocalDaemonServiceEndpoint,
)
from zenml.services.service import BaseService, ServiceConfig
from zenml.services.service_status import ServiceState, ServiceStatus

logger = get_logger(__name__)


SERVICE_DAEMON_CONFIG_FILE_NAME = "service.json"
SERVICE_DAEMON_LOG_FILE_NAME = "service.log"
SERVICE_DAEMON_PID_FILE_NAME = "service.pid"


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
    """

    silent_daemon: Optional[bool] = False
    graceful_shutdown_signal: Optional[signal.Signals] = signal.SIGINT
    forceful_shutdown_signal: Optional[signal.Signals] = signal.SIGKILL


class LocalDaemonServiceStatus(ServiceStatus):
    """Local daemon service status.

    Attributes:
        runtime_path: the path where the service daemon runtime files (the
            configuration file used to start the service daemon and the
            logfile) are located
        silent_daemon: flag indicating whether the output of the daemon
            is suppressed.
    """

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

    @property
    def pid_file(self) -> Optional[str]:
        """Get the path to the daemon PID file where the last known PID of the
        daemon process is stored.

        Returns:
            The path to the PID file, or an empty string, if the
            service has never been started before.
        """
        if not self.runtime_path or self.silent_daemon:
            return ""
        return os.path.join(self.runtime_path, SERVICE_DAEMON_PID_FILE_NAME)

    @property
    def pid(self) -> Optional[int]:
        """Return the PID of the currently running daemon"""
        pid_file = self.pid_file
        if not pid_file:
            return None
        try:
            with open(pid_file, "r") as f:
                pid = int(f.read())
        except OSError:
            return None
        if not pid or not psutil.pid_exists(pid):
            return None
        return pid


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

        if not self.status.pid:
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
            "--pid-file",
            self.status.pid_file,
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

        pid = self.status.pid
        if pid:
            # service daemon is already running
            logger.debug(
                "Daemon process for service '%s' is already running with PID %d",
                self.config.name,
                pid,
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
        p.wait()
        pid = self.status.pid
        if pid:
            logger.debug(
                "Daemon process for service '%s' started with PID: %d",
                self.config.name,
                pid,
            )
        else:
            logger.error(
                "Daemon process for service '%s' failed to start",
                self.config.name,
            )

    def _stop_daemon(self, force: Optional[bool] = False) -> None:
        """Stop the service daemon process associated with this service.

        Args:
            force: if True, the service daemon will be forcefully stopped
        """
        logger.info("Stopping daemon for service '%s' ...", self.config.name)

        pid = self.status.pid
        if not pid:
            # service daemon is not running
            logger.debug(
                "Daemon process for service '%s' no longer running",
                self.config.name,
            )
            return

        s = self.config.graceful_shutdown_signal
        if force:
            s = self.config.forceful_shutdown_signal
        logger.debug(
            "Sending signal %s to daemon process for service '%s' to stop",
            s.name,
            self.config.name,
        )
        os.kill(pid, s)

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
