#  Copyright (c) ZenML GmbH 2021. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at:
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
#  or implied. See the License for the specific language governing
#  permissions and limitations under the License.
import inspect
import json
from abc import abstractmethod
from typing import Any, Dict

from zenml.core.repo import Repository
from zenml.exceptions import PipelineConfigurationError, PipelineInterfaceError
from zenml.logger import get_logger
from zenml.stacks.base_stack import BaseStack
from zenml.utils.analytics_utils import RUN_PIPELINE, track
from zenml.utils.yaml_utils import read_yaml

logger = get_logger(__name__)
PIPELINE_INNER_FUNC_NAME: str = "connect"
PARAM_ENABLE_CACHE: str = "enable_cache"


class BasePipelineMeta(type):
    """Pipeline Metaclass responsible for validating the pipeline definition."""

    def __new__(mcs, name, bases, dct):
        """Ensures that all function arguments are either a `Step`
        or an `Input`."""
        cls = super().__new__(mcs, name, bases, dct)
        cls.NAME = name
        cls.STEP_SPEC = dict()

        connect_spec = inspect.getfullargspec(
            getattr(cls, PIPELINE_INNER_FUNC_NAME)
        )
        connect_args = connect_spec.args

        if connect_args and connect_args[0] == "self":
            connect_args.pop(0)

        for arg in connect_args:
            arg_type = connect_spec.annotations.get(arg, None)
            cls.STEP_SPEC.update({arg: arg_type})
        return cls


class BasePipeline(metaclass=BasePipelineMeta):
    """Base ZenML pipeline."""

    def __init__(self, *args, **kwargs):
        self.__stack = Repository().get_active_stack()
        self.enable_cache = getattr(self, PARAM_ENABLE_CACHE)
        self.pipeline_name = self.__class__.__name__
        self.run_prefix = None
        self.__steps = dict()
        logger.info(f"Creating pipeline: {self.pipeline_name}")
        logger.info(
            f'Cache {"enabled" if self.enable_cache else "disabled"} for '
            f"pipeline `{self.pipeline_name}`"
        )

        if args:
            raise PipelineInterfaceError(
                "You can only use keyword arguments while you are creating an "
                "instance of a pipeline."
            )

        for k, v in kwargs.items():
            if k in self.STEP_SPEC:
                self.__steps.update({k: v})
            else:
                raise PipelineInterfaceError(
                    f"The argument {k} is an unknown argument. Needs to be "
                    f"one of {self.STEP_SPEC.keys()}"
                )

    @abstractmethod
    def connect(self, *args, **kwargs):
        """Function that connects inputs and outputs of the pipeline steps."""

    @property
    def name(self) -> str:
        """Name of pipeline is always equal to self.NAME"""
        return self.NAME

    @property
    def stack(self) -> BaseStack:
        """Returns the stack for this pipeline."""
        return self.__stack

    @stack.setter
    def stack(self, stack: BaseStack):
        """Setting the stack property is not allowed. This method always
        raises a PipelineInterfaceError.
        """
        raise PipelineInterfaceError(
            "The stack will be automatically inferred from your environment. "
            "Please do no attempt to manually change it."
        )

    @property
    def steps(self) -> Dict:
        """Returns a dictionary of pipeline steps."""
        return self.__steps

    @steps.setter
    def steps(self, steps: Dict):
        """Setting the steps property is not allowed. This method always
        raises a PipelineInterfaceError.
        """
        raise PipelineInterfaceError("Cannot set steps manually!")

    @track(event=RUN_PIPELINE)
    def run(self):
        """Runs the pipeline using the orchestrator of the pipeline stack."""
        logger.info(
            f"Using orchestrator `{self.stack.orchestrator_name}` for "
            f"pipeline `{self.pipeline_name}`. Running pipeline.."
        )
        return self.stack.orchestrator.run(self)

    def with_config(
        self, config_file: str, overwrite_step_parameters: bool = False
    ) -> "BasePipeline":
        """Configures this pipeline using a yaml file.

        Args:
            config_file: Path to a yaml file which contains configuration
                options for running this pipeline. See [TODO](url) for details
                regarding the specification of this file.
            overwrite_step_parameters: If set to `True`, values from the
                configuration file will overwrite configuration parameters
                passed in code.

        Returns:
            The pipeline object that this method was called on.
        """
        config_yaml = read_yaml(config_file)

        if "run_prefix" in config_yaml:
            self.run_prefix = config_yaml["run_prefix"]

        if "steps" in config_yaml:
            self._read_config_steps(
                config_yaml["steps"], overwrite=overwrite_step_parameters
            )

        return self

    def _read_config_steps(
        self, steps: Dict[str, Dict[str, Any]], overwrite: bool = False
    ) -> None:
        """Reads and sets step parameters from a config file.

        Args:
            steps: Maps step names to dicts of parameter names and values.
            overwrite: If `True`, overwrite previously set step parameters.
        """
        for step_name, parameters in steps.items():
            if step_name not in self.__steps:
                raise PipelineConfigurationError(
                    f"Found '{step_name}' step in configuration yaml but it "
                    f"doesn't exist in the pipeline steps "
                    f"{list(self.__steps.keys())}."
                )

            step = self.__steps[step_name]
            for parameter, value in parameters.items():
                if parameter not in step.CONFIG.__fields__:
                    raise PipelineConfigurationError(
                        f"Found parameter '{parameter}' for '{step_name}' step "
                        f"in configuration yaml but it doesn't exist in the "
                        f"configuration class `{step.CONFIG}`. Available "
                        f"parameters for this step: "
                        f"{list(step.CONFIG.__fields__.keys())}."
                    )

                # make sure the value gets serialized to a string
                value = json.dumps(value)
                previous_value = step.PARAM_SPEC.get(parameter, None)

                if overwrite:
                    step.PARAM_SPEC[parameter] = value
                else:
                    step.PARAM_SPEC.setdefault(parameter, value)

                if overwrite or not previous_value:
                    logger.debug(
                        "Setting parameter %s=%s for step '%s'.",
                        parameter,
                        value,
                        step_name,
                    )
                if previous_value and not overwrite:
                    logger.warning(
                        "Parameter '%s' from configuration yaml will NOT be "
                        "set as a configuration object was given when "
                        "creating the step. Set `overwrite_step_parameters="
                        "True` when setting the configuration yaml to always "
                        "use the options specified in the yaml file.",
                        parameter,
                    )
