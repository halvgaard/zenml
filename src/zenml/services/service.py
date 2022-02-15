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
from abc import abstractmethod
from typing import Any, ClassVar, Dict, Optional, Tuple, Type, cast
from uuid import UUID, uuid4

from pydantic import Field

from zenml.logger import get_logger
from zenml.services.service_endpoint import BaseServiceEndpoint
from zenml.services.service_registry import ServiceRegistry
from zenml.services.service_status import ServiceState, ServiceStatus
from zenml.services.service_type import ServiceType
from zenml.utils.typed_model import BaseTypedModel, BaseTypedModelMeta
from zenml.utils.yaml_utils import UUIDEncoder

logger = get_logger(__name__)


class ServiceConfig(BaseTypedModel):
    """Generic service configuration.

    Attributes:
        name: unique name for the service instance
        description: description of the service
        uuid: unique identifier for the service
    """

    # TODO: pipeline metadata (name, run id, step etc)
    name: str = ""
    description: str = ""


class BaseServiceMeta(BaseTypedModelMeta):
    """Metaclass responsible for registering different BaseService
    subclasses.

    This metaclass has two main responsibilities:
    1. register all BaseService types in the service registry. This is relevant
    when services are deserialized and instantiated from their JSON or dict
    representation, because their type needs to be known beforehand.
    2. ensuring BaseService instance uniqueness by enforcing that no two
    service instances have the same UUID value. Implementing this at the
    constructor level guarantees that deserializing a service instance from
    a JSON representation multiple times always return the same service object.
    """

    def __new__(
        mcs, name: str, bases: Tuple[Type[Any], ...], dct: Dict[str, Any]
    ) -> "BaseServiceMeta":
        """Creates a BaseService class and registers it in
        the `ServiceRegistry`."""
        cls = cast(Type["BaseService"], super().__new__(mcs, name, bases, dct))
        # register only classes of concrete service implementations
        if not hasattr(cls, "SERVICE_TYPE") or cls.SERVICE_TYPE is None:
            return cls

        # register the service type in the service registry
        ServiceRegistry().register_service_type(cls)
        return cls

    def __call__(cls, *args: Any, **kwargs: Any) -> "BaseServiceMeta":
        """Validate the creation of a service."""
        if not getattr(cls, "SERVICE_TYPE", None):
            raise RuntimeError(
                f"Untyped services instances are not allowed. Please set the "
                f"SERVICE_TYPE class attribute for {cls}."
            )
        uuid = kwargs.get("uuid", None)
        if uuid:
            if isinstance(uuid, str):
                uuid = UUID(uuid)
            if not isinstance(uuid, UUID):
                raise ValueError(
                    f"The `uuid` argument for {cls} must be a UUID instance or a "
                    f"string representation of a UUID."
                )

            # if a service instance with the same UUID is already registered,
            # return the existing instance rather than the newly created one
            existing_service = ServiceRegistry().get_service(uuid)
            if existing_service:
                logger.debug(
                    f"Reusing existing service '{existing_service}' "
                    f"instead of creating a new service with the same UUID."
                )
                return cast("BaseServiceMeta", existing_service)

        svc = cast("BaseService", super().__call__(*args, **kwargs))
        ServiceRegistry().register_service(svc)
        return cast("BaseServiceMeta", svc)


class BaseService(BaseTypedModel, metaclass=BaseServiceMeta):
    """Base service class

    This class implements generic functionality concerning the life-cycle
    management and tracking of an external service (e.g. process, container,
    kubernetes deployment etc.).
    """

    SERVICE_TYPE: ClassVar[ServiceType]

    uuid: UUID = Field(default_factory=uuid4, allow_mutation=False)
    admin_state: ServiceState = ServiceState.INACTIVE
    config: ServiceConfig = Field(default_factory=ServiceConfig)
    status: ServiceStatus = Field(default_factory=ServiceStatus)
    # TODO [MEDIUM] allow multiple endpoints per service
    endpoint: Optional[BaseServiceEndpoint]

    def __init__(
        self,
        **attrs: Any,
    ) -> None:
        super().__init__(**attrs)
        self.config.name = self.config.name or self.__class__.__name__

    @abstractmethod
    def check_status(self) -> Tuple[ServiceState, str]:
        """Check the the current operational state of the external service.

        This method should be overridden by subclasses that implement
        concrete service tracking functionality.

        Returns:
            The operational state of the external service and a message
            providing additional information about that state (e.g. a
            description of the error, if one is encountered while checking the
            service status).
        """

    def update_status(self) -> None:
        """Check the the current operational state of the external service
        and update the local operational status information to reflect it.

        This method should be overridden by subclasses that implement
        concrete service status tracking functionality.
        """
        logger.debug(
            "Running status check for service '%s' ...",
            self,
        )
        state, err = self.check_status()
        logger.debug(
            "Status check results for service '%s': %s [%s]",
            self,
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
        service_dict = self.dict()
        service_dict["service_type"] = self.SERVICE_TYPE.dict()
        return service_dict

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
        return cls.parse_obj(service_dict)

    @classmethod
    def from_json(
        cls,
        json_str: str,
    ) -> "BaseService":
        service_dict = json.loads(json_str)
        return cls.from_dict(service_dict)

    def poll_service_status(self, timeout: int = 0) -> None:
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
                f"Timed out waiting for service {self} to become "
                f"{self.admin_state.value}: {self.status.last_error}"
            )

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

    def deprovision(self, force: bool = False) -> None:
        """Deprovisions all resources used by the service."""
        raise NotImplementedError(
            f"Deprovisioning resources not implemented for {self}."
        )

    def start(self, timeout: int = 0) -> None:
        """Starts the service."""
        self.admin_state = ServiceState.ACTIVE
        self.provision()
        self.poll_service_status(timeout)

    def stop(self, timeout: int = 0, force: bool = False) -> None:
        self.admin_state = ServiceState.INACTIVE
        self.deprovision(force)
        self.poll_service_status(timeout)

    def __repr__(self) -> str:
        """String representation of the service."""
        return f"{self.__class__.__qualname__}[{self.uuid}] (type: {self.SERVICE_TYPE.type}, flavor: {self.SERVICE_TYPE.flavor})"

    def __str__(self) -> str:
        """String representation of the service."""
        return self.__repr__()

    class Config:
        """Pydantic configuration class."""

        # validate attribute assignments
        validate_assignment = True
        # all attributes with leading underscore are private and therefore
        # are mutable and not included in serialization
        underscore_attrs_are_private = True
