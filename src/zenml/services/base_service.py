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

from abc import ABC, ABCMeta, abstractmethod

import json
from pydantic import BaseModel, Field
import time
from typing import Any, Dict, Optional, Tuple, Type, cast

from zenml.services.service_registry import ServiceRegistry
from zenml.services.service_type import ServiceType
from zenml.utils.enum_utils import StrEnum
from zenml.utils.yaml_utils import UUIDEncoder
from zenml.logger import get_logger
from uuid import UUID, uuid4

logger = get_logger(__name__)


class ServiceState(StrEnum):
    """Possible states for the service and service endpoint."""

    ACTIVE = "active"
    PENDING_STARTUP = "pending_startup"
    INACTIVE = "inactive"
    PENDING_SHUTDOWN = "pending_shutdown"
    ERROR = "error"


class ServiceConfig(BaseModel):
    """Generic service configuration.

    Attributes:
        description: description of the service
        uuid: unique identifier for the service
    """

    # TODO: pipeline metadata (name, run id, step etc)
    name: Optional[str]
    description: Optional[str]
    uuid: UUID = Field(default_factory=uuid4)


class ServiceStatus(BaseModel):
    """Information describing the operational status of an external process
    or service tracked by ZenML (e.g. process, container, kubernetes
    deployment etc.).

    Attributes:
        state: the current operational state
        last_state: the operational state prior to the last status update
        last_error: the error encountered during the last status update
    """

    state: Optional[ServiceState] = ServiceState.INACTIVE
    last_state: Optional[ServiceState] = ServiceState.INACTIVE
    last_error: Optional[str]

    def update_state(
        self,
        new_state: Optional[ServiceState] = None,
        error: Optional[str] = None,
    ) -> None:
        """Update the current operational state to reflect a new state
        value and/or error.

        Args:
            new_state: new operational state discovered by the last service
                status update
            error: error message describing an operational failure encountered
                during the last service status update
        """
        if new_state and self.state != new_state:
            self.last_state = self.state
            self.state = new_state
        if error:
            self.last_error = error

    def clear_error(self) -> None:
        """Clear the last error message."""
        self.last_error = None


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

    name: str
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
        # dummy base implementation: assume operational state always
        # matches the admin state
        return endpoint.admin_state, None

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


class BaseServiceEndpoint(ABC):
    """Base service class

    This class implements generic functionality concerning the life-cycle
    management and tracking of an external service endpoint (e.g. a HTTP/HTTPS
    API or generic TCP endpoint exposed by a service).
    """

    STATUS_TYPE = ServiceEndpointStatus
    CONFIG_TYPE = ServiceConfig
    MONITOR_TYPE = BaseServiceEndpointHealthMonitor

    def __init__(
        self,
        config: ServiceEndpointConfig,
        monitor: Optional[BaseServiceEndpointHealthMonitor] = None,
    ) -> None:
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
        state, err = self.check_status()
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


class BaseServiceMeta(ABCMeta):
    """Metaclass responsible for registering different BaseService
    subclasses."""

    def __new__(
        mcs, name: str, bases: Tuple[Type[Any], ...], dct: Dict[str, Any]
    ) -> "BaseServiceMeta":
        """Creates a BaseService class and registers it in
        the `ServiceRegistry`."""
        cls = cast(Type["BaseService"], super().__new__(mcs, name, bases, dct))
        # skip registering abstract classes; only classes of concrete service
        # implementations can be instantiated
        if hasattr(cls.type, "__isabstractmethod__"):
            return cls

        # register the service type in the service registry
        ServiceRegistry().register_service_type(cls)
        return cls


class BaseService(metaclass=BaseServiceMeta):
    """Base service class

    This class implements generic functionality concerning the life-cycle
    management and tracking of an external service (e.g. process, container,
    kubernetes deployment etc.).
    """

    CONFIG_TYPE = ServiceConfig
    STATUS_TYPE = ServiceStatus
    ENDPOINT_TYPE = BaseServiceEndpoint

    def __init__(
        self,
        config: ServiceConfig,
        endpoint: Optional[BaseServiceEndpoint] = None,
    ) -> None:
        self.config = config
        self.status = self.STATUS_TYPE()
        # TODO [LOW]: allow for a service to track multiple endpoints
        self.endpoint = endpoint
        # TODO [LOW]: allow for health monitors to be configured for individual
        #   service endpoints
        self.admin_state = ServiceState.INACTIVE

    @classmethod
    @abstractmethod
    def type(cls) -> ServiceType:
        """The service type.

        Concrete service implementations must override this method and return
        a service type descriptor.
        """

    @abstractmethod
    def check_status(self) -> Tuple[ServiceState, Optional[str]]:
        """Check the the current operational state of the external service.

        This method should be overridden by subclasses that implement
        concrete service tracking functionality.

        Returns:
            The operational state of the external service and an optional error
            message, if an error is encountered while checking the service
            status.
        """
        # dummy base implementation: assume service operational state always
        # matches the admin state
        return self.admin_state, None

    def update_status(self) -> None:
        """Check the the current operational state of the external service
        and update the local operational status information to reflect it.

        This method should be overridden by subclasses that implement
        concrete service status tracking functionality.
        """
        state, err = self.check_status()
        self.status.update_state(state, err)
        if self.endpoint:
            self.endpoint.update_status()

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the service configuration and status in JSON-able format.

        Returns:
            The service config and current status serialized as JSON-able dict.
        """
        return dict(
            type=self.type().dict(),
            config=self.config.dict(),
            status=self.status.dict(),
            endpoint=self.endpoint.to_dict() if self.endpoint else None,
        )

    def to_json(self) -> str:
        """Serialize the service configuration and status in JSON format.

        Returns:
            The service config and current status serialized as JSON.
        """
        return json.dumps(self.to_dict(), indent=4, cls=UUIDEncoder)

    @classmethod
    def from_dict(
        cls,
        service_dict: Dict[str, Any],
    ) -> "BaseService":
        """Instantiate a service instance from a serialized JSON-able
        dict representation containing service and endpoint configuration, and
        an optional last known status.

        Args:
            service_dict: the service and endpoint config and optional last known
            status serialized as JSON-able dict.

        Returns:
            A service instance created from the serialized configuration
            and status.
        """
        config = service_dict.get("config")
        if config is not None:
            config = cls.CONFIG_TYPE.parse_obj(config)
        else:
            config = cls.CONFIG_TYPE()
        status = service_dict.get("status")
        if status is not None:
            status = cls.STATUS_TYPE.parse_obj(status)
        else:
            config = cls.STATUS_TYPE()
        endpoint = service_dict.get("endpoint")
        if endpoint is not None:
            endpoint = cls.ENDPOINT_TYPE.from_dict(endpoint)
        service = cls(config, endpoint)
        service.status = status
        return service

    @classmethod
    def from_json(
        cls,
        json_str: str,
    ) -> "BaseService":
        service_dict = json.loads(json_str)
        return cls.from_dict(service_dict)

    def poll_service_status(self, timeout: int) -> None:
        """Poll the external service status until the service operational
        state matches the administrative state, or the timeout is reached.

        Args:
            timeout: maximum time to wait for the service operational state
            to match the administrative state, in seconds
        """
        time_remaining = timeout
        while True:
            if self.admin_state == ServiceState.ACTIVE and self.is_running:
                return
            if self.admin_state == ServiceState.INACTIVE and self.is_stopped:
                return
            if time_remaining <= 0:
                break
            time.sleep(1)
            time_remaining -= 1

        if timeout > 0:
            logger.error(
                f"Timed out waiting for service to become {self.admin_state.value}"
                f": {self.status.last_error}"
            )

    def prepare_deployment(
        self,
    ) -> None:
        """Prepares deploying the service.

        This method gets called immediately before the service is deployed.
        Subclasses should override it if they need to run code before the
        service deployment.
        """

    def prepare_run(self) -> None:
        """Prepares running the service."""

    def cleanup_run(self) -> None:
        """Cleans up resources after the service run is finished."""

    @property
    def is_provisioned(self) -> bool:
        """If the service provisioned resources."""
        return True

    @property
    def is_running(self) -> bool:
        self.update_status()
        return self.status.state == ServiceState.ACTIVE and (
            not self.endpoint or self.endpoint.is_active()
        )

    @property
    def is_stopped(self) -> bool:
        self.update_status()
        return self.status.state == ServiceState.INACTIVE

    def provision(self) -> None:
        """Provisions resources to run the service."""
        raise NotImplementedError(
            f"Provisioning resources not implemented for {self}."
        )

    def deprovision(self) -> None:
        """Deprovisions all resources used by the service after the service is
        shut down."""
        raise NotImplementedError(
            f"Deprovisioning resources not implemented for {self}."
        )

    def resume(self) -> None:
        """Resumes the service."""
        raise NotImplementedError(
            f"Resume operation not implemented for service {self}."
        )

    def suspend(self) -> None:
        """Suspends the service."""
        raise NotImplementedError(
            f"Suspend operation not implemented for service {self}."
        )

    def start(self, timeout: int) -> None:
        self.admin_state = ServiceState.ACTIVE
        self.provision()
        self.poll_service_status(timeout)

    def stop(self, timeout: int, force: Optional[bool] = False) -> None:
        self.admin_state = ServiceState.INACTIVE
        self.deprovision(force)
        self.poll_service_status(timeout)

    def __repr__(self) -> str:
        """String representation of the service."""
        _repr = self.to_json()
        return (
            f"{self.__class__.__qualname__}(type={self.type}, "
            f"flavor={self.flavor}, {_repr})"
        )

    def __str__(self) -> str:
        """String representation of the service."""
        return self.__repr__()
