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

from abc import ABC

import json
from pydantic import BaseModel
from typing import Any, Dict, Optional, Tuple

from zenml.logger import get_logger
from zenml.services.service_monitor import BaseServiceEndpointHealthMonitor
from zenml.services.service_status import ServiceState, ServiceStatus
from zenml.utils.enum_utils import StrEnum

logger = get_logger(__name__)


class ServiceEndpointProtocol(StrEnum):
    """Possible endpoint protocol values."""

    TCP = "tcp"
    HTTP = "http"
    HTTPS = "https"


class ServiceEndpointConfig(BaseModel):
    """Generic service endpoint configuration.

    Attributes:
        name: unique name for the service endpoint
        description: description of the service endpoint
    """

    name: Optional[str]
    description: Optional[str]


class ServiceEndpointStatus(ServiceStatus):
    """Status information describing the operational state of a service
    endpoint (e.g. a HTTP/HTTPS API or generic TCP endpoint exposed by a
    service).

    Attributes:
        protocol: the TCP protocol used by the service endpoint
        hostname: the hostname where the service endpoint is accessible
        port: the current TCP port where the service endpoint is accessible
    """

    protocol: Optional[ServiceEndpointProtocol] = ServiceEndpointProtocol.TCP
    hostname: Optional[str]
    port: Optional[int]

    @property
    def uri(self) -> Optional[str]:
        """Get the URI of the service endpoint.

        Returns:
            The URI of the service endpoint or None, if the service endpoint
            operational status doesn't have the required information.
        """
        if not self.hostname or not self.port:
            # the service is not yet in a state in which the endpoint hostname
            # and port are known
            return None

        return f"{self.protocol.value}://{self.hostname}:{self.port}"


class BaseServiceEndpoint(ABC):
    """Base service class

    This class implements generic functionality concerning the life-cycle
    management and tracking of an external service endpoint (e.g. a HTTP/HTTPS
    API or generic TCP endpoint exposed by a service).
    """

    STATUS_TYPE = ServiceEndpointStatus
    CONFIG_TYPE = ServiceEndpointConfig
    MONITOR_TYPE = BaseServiceEndpointHealthMonitor

    def __init__(
        self,
        config: ServiceEndpointConfig,
        monitor: Optional[BaseServiceEndpointHealthMonitor] = None,
    ) -> None:
        config.name = config.name or self.__class__.__name__
        self.config = config
        self.monitor = monitor
        self.status = self.STATUS_TYPE()
        self.admin_state = ServiceState.INACTIVE

    def check_status(self) -> Tuple[ServiceState, Optional[str]]:
        """Check the the current operational state of the external
        service endpoint.

        Returns:
            The operational state of the external service endpoint and an
            optional error message, if an error is encountered while checking
            the service endpoint status.
        """
        if not self.monitor:
            # no health monitor configured; assume service operational state
            # always matches the admin state
            return self.admin_state, None
        return self.monitor.check_endpoint_status(self)

    def update_status(self) -> None:
        """Check the the current operational state of the external service
        endpoint and update the local operational status information to reflect
        it.
        """
        logger.debug(
            "Running health check for service endpoint '%s' ...",
            self.config.name,
        )
        state, err = self.check_status()
        logger.debug(
            "Health check results for service endpoint '%s': %s [%s]",
            self.config.name,
            state.name,
            err,
        )
        self.status.update_state(state, err)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the service endpoint configuration and status in JSON-able
        format.

        Returns:
            The service endpoint config and current status serialized as
            JSON-able dict.
        """
        return dict(
            config=self.config.dict(),
            monitor=self.monitor.to_dict() if self.monitor else None,
            status=self.status.dict(),
        )

    def to_json(self) -> str:
        """Serialize the service endpoint configuration and status in JSON
        format.

        Returns:
            The service endpoint config and current status serialized as JSON.
        """
        return json.dumps(self.to_dict(), indent=4)

    @classmethod
    def from_dict(
        cls,
        endpoint_dict: Dict[str, Any],
    ) -> "BaseServiceEndpoint":
        """Instantiate a service endpoint instance from a serialized JSON-able
        dict representation containing service endpoint configuration and an
        optional last known status.

        Args:
            endpoint_dict: the service endpoint config and optional last known
            status serialized as JSON-able dict.

        Returns:
            A service endpoint instance created from the serialized configuration
            and status.
        """
        config = endpoint_dict.get("config")
        if config is not None:
            config = cls.CONFIG_TYPE.parse_obj(config)
        else:
            config = cls.CONFIG_TYPE()
        status = endpoint_dict.get("status")
        if status is not None:
            status = cls.STATUS_TYPE.parse_obj(status)
        else:
            config = cls.STATUS_TYPE()
        monitor = endpoint_dict.get("monitor")
        if monitor is not None:
            monitor = cls.MONITOR_TYPE.from_dict(monitor)
        endpoint = cls(config, monitor)
        endpoint.status = status
        return endpoint

    @classmethod
    def from_json(
        cls,
        json_str: str,
    ) -> "BaseServiceEndpoint":
        endpoint_dict = json.loads(json_str)
        return cls.from_dict(endpoint_dict)

    def is_active(self) -> bool:
        self.update_status()
        return self.status.state == ServiceState.ACTIVE

    def is_inactive(self) -> bool:
        self.update_status()
        return self.status.state == ServiceState.INACTIVE
