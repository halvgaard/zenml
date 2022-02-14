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
import time
from abc import ABCMeta, abstractmethod
from typing import Any, Dict, Optional, Tuple, Type, cast
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from zenml.logger import get_logger
from zenml.services.service_endpoint import BaseServiceEndpoint
from zenml.services.service_registry import ServiceRegistry
from zenml.services.service_status import ServiceState, ServiceStatus
from zenml.services.service_type import ServiceType
from zenml.utils.yaml_utils import UUIDEncoder

logger = get_logger(__name__)


class ServiceConfig(BaseModel):
    """Generic service configuration.

    Attributes:
        name: unique name for the service instance
        description: description of the service
        uuid: unique identifier for the service
    """

    # TODO: pipeline metadata (name, run id, step etc)
    name: Optional[str]
    description: Optional[str]
    uuid: UUID = Field(default_factory=uuid4)


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
        config.name = config.name or self.__class__.__name__
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

    def update_status(self) -> None:
        """Check the the current operational state of the external service
        and update the local operational status information to reflect it.

        This method should be overridden by subclasses that implement
        concrete service status tracking functionality.
        """
        logger.debug(
            "Running status check for service '%s' ...",
            self.config.name,
        )
        state, err = self.check_status()
        logger.debug(
            "Status check results for service '%s': %s [%s]",
            self.config.name,
            state.name,
            err,
        )
        self.status.update_state(state, err)

        # don't bother checking the endpoint state if the service is not active
        if self.status.state == ServiceState.INACTIVE:
            return

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

    def deprovision(self, force: Optional[bool] = False) -> None:
        """Deprovisions all resources used by the service."""
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
        """Starts the service."""
        if not ServiceRegistry().service_is_registered(self.config.uuid):
            ServiceRegistry().register_service(self)
        self.admin_state = ServiceState.ACTIVE
        self.provision()
        self.poll_service_status(timeout)

    def stop(self, timeout: int, force: Optional[bool] = False) -> None:
        self.admin_state = ServiceState.INACTIVE
        self.deprovision(force)
        self.poll_service_status(timeout)

    def __repr__(self) -> str:
        """String representation of the service."""
        service_type = self.type()
        return (
            f"{self.__class__.__qualname__}(type={service_type.type}, "
            f"flavor={service_type.flavor}, uuid={self.config.uuid})"
        )

    def __str__(self) -> str:
        """String representation of the service."""
        return self.__repr__()
