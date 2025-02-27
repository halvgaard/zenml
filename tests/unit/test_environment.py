#  Copyright (c) ZenML GmbH 2021. All Rights Reserved.
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

import platform

import pytest

from zenml.constants import VALID_OPERATING_SYSTEMS
from zenml.environment import BaseEnvironmentComponent, Environment
from zenml.steps import StepEnvironment


def test_environment_platform_info_correctness():
    """Checks that `Environment.get_system_info()` returns the correct
    platform"""
    system_id = platform.system()

    if system_id == "Darwin":
        system_id = "mac"
    elif system_id not in VALID_OPERATING_SYSTEMS:
        system_id = "unknown"

    assert system_id.lower() == Environment.get_system_info()["os"]


def test_environment_is_singleton():
    """Tests that environment is a singleton."""
    assert Environment() is Environment()


def test_step_is_running():
    """Tests that the environment correctly reports when a step is running."""

    assert Environment().step_is_running is False

    with StepEnvironment(
        pipeline_name="pipeline",
        pipeline_run_id="run_id",
        step_name="step",
    ):
        assert Environment().step_is_running is True

    assert Environment().step_is_running is False


def test_environment_component_activation():
    """Tests that environment components can be activated and deactivated."""

    class Foo(BaseEnvironmentComponent):
        NAME = "foo"

    assert Environment().get_component("foo") is None
    assert not Environment().has_component("foo")
    with pytest.raises(KeyError):
        Environment()["foo"]

    f = Foo()
    assert f.active is False

    with f:
        assert f.active is True
        assert Environment().get_component("foo") is f
        assert Environment().has_component("foo")
        assert Environment()["foo"] is f

    assert f.active is False
    assert Environment().get_component("foo") is None
    assert not Environment().has_component("foo")
    with pytest.raises(KeyError):
        Environment()["foo"]
