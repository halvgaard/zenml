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

import json
import socket
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

import requests  # type: ignore
from pydantic import BaseModel

from zenml.logger import get_logger
from zenml.services.service_status import ServiceState

logger = get_logger(__name__)


if TYPE_CHECKING:
    from zenml.services.service_endpoint import BaseServiceEndpoint


DEFAULT_HTTP_HEALTHCHECK_TIMEOUT = 5


class ServiceEndpointHealthMonitorConfig(BaseModel):
    """Generic service health monitor configuration."""


class BaseServiceEndpointHealthMonitor(ABC):
    """Base class used for service endpoint health monitors."""

    CONFIG_TYPE = ServiceEndpointHealthMonitorConfig

    def __init__(
        self,
        config: ServiceEndpointHealthMonitorConfig,
    ) -> None:
        self.config = config

    @abstractmethod
    def check_endpoint_status(
        self, endpoint: "BaseServiceEndpoint"
    ) -> Tuple[ServiceState, Optional[str]]:
        """Check the the current operational state of the external
        service endpoint.

        This method should be overridden by subclasses that implement
        concrete service endpoint tracking functionality.

        Returns:
            The operational state of the external service endpoint and an
            optional error message, if an error is encountered while checking
            the service endpoint status.
        """

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the endpoint monitor configuration in JSON-able
        format.

        Returns:
            The endpoint monitor config serialized as JSON-able dict.
        """
        return dict(
            config=self.config.dict(),
        )

    def to_json(self) -> str:
        """Serialize the endpoint monitor configuration in JSON format.

        Returns:
            The endpoint monitor config serialized as JSON.
        """
        return json.dumps(self.to_dict(), indent=4)

    @classmethod
    def from_dict(
        cls,
        monitor_dict: Dict[str, Any],
    ) -> "BaseServiceEndpointHealthMonitor":
        """Instantiate an endpoint monitor instance from a serialized JSON-able
        dict representation containing monitor configuration.

        Args:
            monitor_dict: the endpoint monitor config serialized as JSON-able
                dict.

        Returns:
            An endpoint monitor instance created from the serialized
            configuration.
        """
        config = monitor_dict.get("config")
        if config is not None:
            config = cls.CONFIG_TYPE.parse_obj(config)
        else:
            config = cls.CONFIG_TYPE()
        return cls(config)

    @classmethod
    def from_json(
        cls,
        json_str: str,
    ) -> "BaseServiceEndpointHealthMonitor":
        monitor_dict = json.loads(json_str)
        return cls.from_dict(monitor_dict)


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
        self.config = config

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
        if not endpoint.status.port or not endpoint.status.hostname:
            return ServiceState.ERROR, "TCP port and hostname values are not known"

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
