#  Copyright (c) ZenML GmbH 2022. All Rights Reserved.
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

import mlflow
import numpy as np
import os
import tensorflow as tf

from zenml.environment import Environment
from zenml.integrations.mlflow.mlflow_step_decorator import enable_mlflow
from zenml.integrations.mlflow.mlflow_environment import (
    MLFLOW_STEP_ENVIRONMENT_NAME,
)

from zenml.pipelines import pipeline
from zenml.steps import BaseStepConfig, Output, step

from zenml.integrations.mlflow.services.mlflow_deployment import (
    MLFlowDeploymentService,
    MLFlowDeploymentConfig,
)

# Path to a pip requirements file that contains requirements necessary to run
# the pipeline
requirements_file = os.path.join(os.path.dirname(__file__), "requirements.txt")


class TrainerConfig(BaseStepConfig):
    """Trainer params"""

    epochs: int = 1
    lr: float = 0.001


@step
def importer_mnist() -> Output(
    x_train=np.ndarray, y_train=np.ndarray, x_test=np.ndarray, y_test=np.ndarray
):
    """Download the MNIST data store it as an artifact"""
    (x_train, y_train), (
        x_test,
        y_test,
    ) = tf.keras.datasets.mnist.load_data()
    return x_train, y_train, x_test, y_test


@step
def normalizer(
    x_train: np.ndarray, x_test: np.ndarray
) -> Output(x_train_normed=np.ndarray, x_test_normed=np.ndarray):
    """Normalize the values for all the images so they are between 0 and 1"""
    x_train_normed = x_train / 255.0
    x_test_normed = x_test / 255.0
    return x_train_normed, x_test_normed


# Define the step and enable mlflow - order of decorators is important here
@enable_mlflow
@step
def tf_trainer(
    config: TrainerConfig,
    x_train: np.ndarray,
    y_train: np.ndarray,
) -> tf.keras.Model:
    """Train a neural net from scratch to recognize MNIST digits return our
    model or the learner"""
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(10),
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(config.lr),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    mlflow.tensorflow.autolog()
    model.fit(
        x_train,
        y_train,
        epochs=config.epochs,
    )

    # write model
    return model


# Define the step and enable mlflow - order of decorators is important here
@enable_mlflow
@step
def tf_evaluator(
    x_test: np.ndarray,
    y_test: np.ndarray,
    model: tf.keras.Model,
) -> float:
    """Calculate the loss for the model for each epoch in a graph"""

    _, test_acc = model.evaluate(x_test, y_test, verbose=2)
    mlflow.log_metric("val_accuracy", test_acc)
    return test_acc


# Define the step and enable mlflow - order of decorators is important here
@enable_mlflow
@step
def predictor(model: tf.keras.Model) -> MLFlowDeploymentService:
    """Start a prediction service

    NOTE: the input argument is just a dummy that allows us to enforce that
    the prediction step is executed after the training step, otherwise no
    model is found in the current MLflow run.
    """

    predictor_cfg = MLFlowDeploymentConfig(
        model_uri=mlflow.get_artifact_uri("model"),
        workers=3,
        mlserver=False,
    )

    service = MLFlowDeploymentService(
        config=predictor_cfg,
    )

    service.start(timeout=10)

    return service


@step
def batch_inference(
    service: MLFlowDeploymentService,
    batch: np.ndarray,
) -> Output(predictions=np.ndarray):
    """Run a batch inference request against a prediction service"""

    # service.start(timeout=10) # should be a NOP but isn't currently

    predictions = service.predict(batch)
    predictions = predictions.argmax(axis=-1)
    service.stop(timeout=10)

    return predictions


@step
def inference_evaluator(
    predictions: np.ndarray,
    expectations: np.ndarray,
) -> float:
    """Evaluate the predictions against the expectations"""
    comparison = predictions == expectations
    return np.count_nonzero(comparison) / len(comparison)


@pipeline(enable_cache=False, requirements_file=requirements_file)
def mlflow_example_pipeline(
    importer,
    normalizer,
    trainer,
    evaluator,
    predictor,
    batch_inference,
    inference_evaluator,
):
    # Link all the steps artifacts together
    x_train, y_train, x_test, y_test = importer()
    x_trained_normed, x_test_normed = normalizer(x_train=x_train, x_test=x_test)
    model = trainer(x_train=x_trained_normed, y_train=y_train)
    evaluator(x_test=x_test_normed, y_test=y_test, model=model)
    service = predictor(model)
    predictions = batch_inference(service, x_test)
    inference_evaluator(predictions, y_test)
