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
import shutil
from os import environ
from pathlib import Path
from typing import Dict

import pytest

from zenml.cli import EXAMPLES_RUN_SCRIPT, SHELL_EXECUTABLE, LocalExample
from zenml.enums import ExecutionStatus
from zenml.repository import Repository

QUICKSTART = "quickstart"
NOT_SO_QUICKSTART = "not_so_quickstart"
CACHING = "caching"
DRIFT_DETECTION = "drift_detection"
MLFLOW = "mlflow_tracking"
CUSTOM_MATERIALIZER = "custom_materializer"
WHYLOGS = "whylogs"
FETCH_HISTORICAL_RUNS = "fetch_historical_runs"


@pytest.fixture
def examples_dir(clean_repo):
    # TODO [high]: tests should store zenml artifacts in a new temp directory
    examples_path = Path(clean_repo.root) / "zenml_examples"
    source_path = Path(clean_repo.original_cwd) / "examples"
    shutil.copytree(source_path, examples_path)
    yield examples_path


def example_runner(examples_dir):
    """Get the executable that runs examples.

    By default returns the path to an executable .sh file in the
    repository, but can also prefix that with the path to a shell
    / interpreter when the file is not executable on its own. The
    latter option is needed for windows compatibility.
    """
    return (
        [environ[SHELL_EXECUTABLE]] if SHELL_EXECUTABLE in environ else []
    ) + [str(examples_dir / EXAMPLES_RUN_SCRIPT)]


def test_run_quickstart(examples_dir: Path):
    """Testing the functionality of the quickstart example

    Args:
        examples_dir: Temporary folder containing all examples including the run_examples
        bash script.
    """
    local_example = LocalExample(examples_dir / QUICKSTART, name=QUICKSTART)

    local_example.run_example(example_runner(examples_dir), force=True)

    # Verify the example run was successful
    repo = Repository(local_example.path)
    pipeline = repo.get_pipelines()[0]
    assert pipeline.name == "mnist_pipeline"

    pipeline_run = pipeline.runs[-1]

    assert pipeline_run.status == ExecutionStatus.COMPLETED

    for step in pipeline_run.steps:
        assert step.status == ExecutionStatus.COMPLETED


def test_run_not_so_quickstart(examples_dir: Path):
    """Testing the functionality of the not_so_quickstart example

    Args:
        examples_dir: Temporary folder containing all examples including the run_examples
        bash script.
    """
    local_example = LocalExample(
        examples_dir / NOT_SO_QUICKSTART, name=NOT_SO_QUICKSTART
    )
    local_example.run_example(example_runner(examples_dir), force=True)

    # Verify the example run was successful
    repo = Repository(local_example.path)
    pipeline = repo.get_pipelines()[0]
    assert pipeline.name == "mnist_pipeline"

    first_run = pipeline.runs[-3]
    second_run = pipeline.runs[-2]
    third_run = pipeline.runs[-1]

    assert first_run.status == ExecutionStatus.COMPLETED
    assert second_run.status == ExecutionStatus.COMPLETED
    assert third_run.status == ExecutionStatus.COMPLETED


def test_run_drift_detection(examples_dir: Path):
    """Testing the functionality of the drift_detection example

    Args:
        examples_dir: Temporary folder containing all examples including the run_examples
        bash script.
    """
    local_example = LocalExample(
        examples_dir / DRIFT_DETECTION, name=DRIFT_DETECTION
    )

    local_example.run_example(example_runner(examples_dir), force=True)

    # Verify the example run was successful
    repo = Repository(local_example.path)
    pipeline = repo.get_pipelines()[0]
    assert pipeline.name == "drift_detection_pipeline"

    run = pipeline.runs[0]

    # Run should be completed
    assert run.status == ExecutionStatus.COMPLETED

    # The first run should not have any cached steps
    for step in run.steps:
        assert not step.is_cached

    # Final step should have output a data drift report
    output_obj = run.steps[3].outputs["profile"].read()
    assert isinstance(output_obj, Dict)
    assert output_obj.get("data_drift") is not None


def test_run_caching(examples_dir: Path):
    """Testing the functionality of the caching example

    Args:
        examples_dir: Temporary folder containing all examples including the run_examples
        bash script.
    """
    local_example = LocalExample(examples_dir / CACHING, name=CACHING)
    local_example.run_example(example_runner(examples_dir), force=True)

    # Verify the example run was successful
    repo = Repository(local_example.path)
    pipeline = repo.get_pipelines()[0]
    assert pipeline.name == "mnist_pipeline"

    first_run = pipeline.runs[-2]
    second_run = pipeline.runs[-1]

    # Both runs should be completed
    assert first_run.status == ExecutionStatus.COMPLETED
    assert second_run.status == ExecutionStatus.COMPLETED

    # The first run should not have any cached steps
    for step in first_run.steps:
        assert not step.is_cached

    # The second run should have two cached steps (chronologically first 2)
    assert second_run.steps[0].is_cached
    assert second_run.steps[1].is_cached
    assert not second_run.steps[2].is_cached
    assert not second_run.steps[3].is_cached


def test_run_mlflow(examples_dir: Path):
    """Testing the functionality of the quickstart example

    Args:
        examples_dir: Temporary folder containing all examples including the run_examples
        bash script.
    """
    local_example = LocalExample(examples_dir / MLFLOW, name=MLFLOW)
    local_example.run_example(example_runner(examples_dir), force=True)

    # Verify the example run was successful
    repo = Repository(local_example.path)
    pipeline = repo.get_pipelines()[0]
    assert pipeline.name == "mlflow_example_pipeline"

    first_run = pipeline.runs[-2]
    second_run = pipeline.runs[-1]

    # Both runs should be completed
    assert first_run.status == ExecutionStatus.COMPLETED
    assert second_run.status == ExecutionStatus.COMPLETED

    for step in first_run.steps:
        assert step.status == ExecutionStatus.COMPLETED
    for step in second_run.steps:
        assert step.status == ExecutionStatus.COMPLETED

    import mlflow
    from mlflow.tracking import MlflowClient

    from zenml.integrations.mlflow.mlflow_environment import MLFlowEnvironment

    # Create and activate the global MLflow environment
    MLFlowEnvironment(local_example.path).activate()

    # fetch the MLflow experiment created for the pipeline runs
    mlflow_experiment = mlflow.get_experiment_by_name(pipeline.name)
    assert mlflow_experiment is not None

    # fetch all MLflow runs created for the pipeline
    mlflow_runs = mlflow.search_runs(
        experiment_ids=[mlflow_experiment.experiment_id], output_format="list"
    )
    assert len(mlflow_runs) == 2

    # fetch the MLflow run created for the first pipeline run
    mlflow_runs = mlflow.search_runs(
        experiment_ids=[mlflow_experiment.experiment_id],
        filter_string=f'tags.mlflow.runName = "{first_run.name}"',
        output_format="list",
    )
    assert len(mlflow_runs) == 1
    first_mlflow_run = mlflow_runs[0]

    # fetch the MLflow run created for the second pipeline run
    mlflow_runs = mlflow.search_runs(
        experiment_ids=[mlflow_experiment.experiment_id],
        filter_string=f'tags.mlflow.runName = "{second_run.name}"',
        output_format="list",
    )
    assert len(mlflow_runs) == 1
    second_mlflow_run = mlflow_runs[0]

    client = MlflowClient()
    # fetch the MLflow artifacts logged during the first pipeline run
    artifacts = client.list_artifacts(first_mlflow_run.info.run_id)
    assert len(artifacts) == 3

    # fetch the MLflow artifacts logged during the second pipeline run
    artifacts = client.list_artifacts(second_mlflow_run.info.run_id)
    assert len(artifacts) == 3


def test_whylogs_profiling(examples_dir: Path):
    """Testing the functionality of the whylogs example

    Args:
        examples_dir: Temporary folder containing all examples including the run_examples
        bash script.
    """
    local_example = LocalExample(examples_dir / WHYLOGS, name=WHYLOGS)

    local_example.run_example(example_runner(examples_dir), force=True)

    # Verify the example run was successful
    repo = Repository(local_example.path)
    pipeline = repo.get_pipelines()[0]
    assert pipeline.name == "data_profiling_pipeline"

    run = pipeline.runs[0]

    # Run should be completed
    assert run.status == ExecutionStatus.COMPLETED

    # The first run should not have any cached steps
    for step in run.steps:
        assert not step.is_cached

    from whylogs import DatasetProfile

    # First step should have output a whylogs dataset profile
    output_obj = run.get_step("data_loader").outputs["profile"].read()
    assert isinstance(output_obj, DatasetProfile)

    # Second and third step should also have output a whylogs dataset profile
    output_obj = run.get_step("train_data_profiler").output.read()
    assert isinstance(output_obj, DatasetProfile)
    output_obj = run.get_step("test_data_profiler").output.read()
    assert isinstance(output_obj, DatasetProfile)


def test_run_custom_materializer(examples_dir: Path):
    """Testing the functionality of the custom materializer example.

    Args:
        examples_dir: Temporary folder containing all examples including the
                      run_examples bash script.
    """
    local_example = LocalExample(
        examples_dir / CUSTOM_MATERIALIZER, name=CUSTOM_MATERIALIZER
    )
    local_example.run_example(example_runner(examples_dir), force=True)

    # Verify the example run was successful
    repo = Repository(local_example.path)
    pipeline = repo.get_pipelines()[0]
    first_run = pipeline.runs[-1]

    # Both runs should be completed
    assert first_run.status == ExecutionStatus.COMPLETED


def test_run_fetch_historical_runs(examples_dir: Path):
    """Testing the functionality of the fetch_historical_runs example.

    Args:
        examples_dir: Temporary folder containing all examples including the
                      run_examples bash script.
    """
    local_example = LocalExample(
        examples_dir / FETCH_HISTORICAL_RUNS, name=FETCH_HISTORICAL_RUNS
    )

    local_example.run_example(example_runner(examples_dir), force=True)

    # Verify the example run was successful
    repo = Repository(local_example.path)
    pipeline = repo.get_pipelines()[0]
    assert pipeline.name == "mnist_pipeline"

    pipeline_run = pipeline.runs[-1]

    assert pipeline_run.status == ExecutionStatus.COMPLETED

    for step in pipeline_run.steps:
        assert step.status == ExecutionStatus.COMPLETED
