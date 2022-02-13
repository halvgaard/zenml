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


import socket

from typing import Optional

from zenml.services.service_endpoint import (
    BaseServiceEndpoint,
    ServiceEndpointProtocol,
    ServiceEndpointConfig,
    ServiceEndpointStatus,
)

from zenml.services.service_monitor import (
    HttpEndpointHealthMonitor,
)

from zenml.logger import get_logger

logger = get_logger(__name__)

SERVICE_PORT_RANGE = (8000, 65535)


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
