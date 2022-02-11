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
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Optional,
    Type,
)

from zenml.logger import get_logger
from zenml.services.service_type import ServiceType
from zenml.utils.singleton import SingletonMetaClass
from uuid import UUID

logger = get_logger(__name__)

if TYPE_CHECKING:
    from zenml.services.base_service import BaseService


class ServiceRegistry(metaclass=SingletonMetaClass):
    """Registry of service types and service instances."""

    def __init__(self) -> None:
        self.service_types: Dict[ServiceType, Type["BaseService"]] = {}
        self.services: Dict[UUID, "BaseService"] = {}

    def register_service_type(self, cls: Type["BaseService"]) -> None:
        """Registers a new service type.

        Args:
            cls: A BaseService subclass.
        """
        service_type = cls.type()
        if service_type not in self.service_types:
            self.service_types[service_type] = cls
            logger.debug(
                f"Registered service class {cls} for "
                f"service type <{service_type}>"
            )
        else:
            raise Exception(
                f"Found existing service type for {service_type}: "
                f"{self.service_types[service_type]}. Skipping registration "
                f"of {cls}."
            )

    def get_service_type(
        self, service_type: ServiceType
    ) -> Optional[Type["BaseService"]]:
        """Get the service class registered for a service type.

        Args:
            service_type: service type.

        Returns:
            `BaseService` subclass that was registered for the service type or
            None, if no service class was registered for the service type.
        """
        # Check whether the type is registered
        return self.service_types.get(service_type)

    def get_service_types(
        self,
    ) -> Dict[ServiceType, Type["BaseService"]]:
        """Get all registered service types."""
        return self.service_types

    def service_type_is_registered(self, service_type: ServiceType) -> bool:
        """Check if a service type is registered."""
        return service_type in self.service_types

    def register_service(self, service: "BaseService") -> None:
        """Registers a new service instance

        Args:
            service: A BaseService instance.
        """
        service_type = service.type()
        if service_type not in self.service_types:
            raise TypeError("Service type <{service_type}> is not registered.")

        if service.config.uuid not in self.services:
            self.services[service.config.uuid] = service
            logger.debug(
                f"Registered service '{service.config.name}' of service type "
                f"<{service_type}> with UUID {service.config.uuid}"
            )
        else:
            existing_service = self.services[service.config.uuid]
            raise Exception(
                f"Found existing service '{existing_service.config.name}' and "
                f"service type {existing_service.type()} for UUID: "
                f"{service.config.uuid}. Skipping registration for service "
                f"'{service.config.name}' of type {service_type}."
            )

    def get_service(self, uuid: UUID) -> Optional[Type["BaseService"]]:
        """Get the service instance registered for a UUID.

        Args:
            UUID: service instance identifier.

        Returns:
            `BaseService` instance that was registered for the UUID or
            None, if no matching service instance was found.
        """
        return self.services.get(uuid)

    def service_is_registered(self, uuid: UUID) -> bool:
        """Check if a service instance is registered."""
        return uuid in self.services

    def load_service_from_dict(
        self, service_dict: Dict[str, Any]
    ) -> "BaseService":
        """Load a service instance from its dict representation.

        Creates, registers and returns a service instantiated from the dict
        representation of the service configuration and last known status
        information.

        If an existing service instance with the same UUID is already
        present in the service registry, it is returned instead.

        Args:
            service_dict: dict representation of the service configuration and
                last known status

        Returns:
            A new or existing ZenML service instance.
        """
        service_type = service_dict.get("type")
        if not service_type:
            raise ValueError(
                "Service type not present in the service dictionary"
            )
        service_type = ServiceType.parse_obj(service_type)
        service_class = self.get_service_type(service_type)
        if not service_class:
            raise TypeError(f"Unknown service type: {str(service_type)}")
        service = service_class.from_dict(service_dict)
        existing_service = self.get_service(service.config.uuid)
        if existing_service:
            # TODO: raise error if not the same type
            logger.debug(
                f"Reusing existing service '{existing_service.config.name}' "
                f"of type {existing_service.type()} for UUID: "
                f"{service.config.uuid}."
            )
            return existing_service
        self.register_service(service)
        return service

    def load_service_from_json(self, json_str: str) -> "BaseService":
        """Load a service instance from its JSON representation.

        Creates and returns a service instantiated from the JSON serialized
        service configuration and last known status information.

        Args:
            json_str: JSON string representation of the service configuration
                and last known status

        Returns:
            A ZenML service instance.
        """
        service_dict = json.loads(json_str)
        return self.load_service_from_dict(service_dict)
