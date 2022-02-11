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

from abc import ABC, abstractmethod

import json
from pydantic import BaseModel
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

from zenml.logger import get_logger
from zenml.services.service_status import ServiceState

logger = get_logger(__name__)


if TYPE_CHECKING:
    from zenml.services.service_endpoint import BaseServiceEndpoint


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
