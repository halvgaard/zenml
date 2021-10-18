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

import os

import numpy as np
import tensorflow as tf

from zenml.pipelines import pipeline
from zenml.steps import step
from zenml.steps.base_step_config import BaseStepConfig
from zenml.steps.step_output import Output


class TrainerConfig(BaseStepConfig):
    """Trainer params"""

    epochs: int = 1


@step
def importer_mnist() -> Output(
    X_train=np.ndarray, y_train=np.ndarray, X_test=np.ndarray, y_test=np.ndarray
):
    """Download the MNIST data store it as an artifact"""
    (X_train, y_train), (
        X_test,
        y_test,
    ) = tf.keras.datasets.mnist.load_data()
    return X_train, y_train, X_test, y_test


@step
def normalizer(
    X_train: np.ndarray, X_test: np.ndarray
) -> Output(X_train_normed=np.ndarray, X_test_normed=np.ndarray):
    """Normalize the values for all the images so they are between 0 and 1"""
    X_train_normed = X_train / 255.0
    X_test_normed = X_test / 255.0
    return X_train_normed, X_test_normed


@step
def trainer(
    config: TrainerConfig,
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> tf.keras.Model:
    """Train a neural net from scratch to recognise MNIST digits return our
    model or the learner"""
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(10, activation="relu"),
            tf.keras.layers.Dense(10),
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    model.fit(
        X_train,
        y_train,
        epochs=config.epochs,
    )

    # write model
    return model


@step
def evaluator(
    X_test: np.ndarray,
    y_test: np.ndarray,
    model: tf.keras.Model,
) -> np.ndarray:
    """Calculate the loss for the model for each epoch in a graph"""

    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
    return np.array([test_loss, test_acc])


# Define the pipeline


@pipeline
def mnist_pipeline(
    importer,
    normalizer: normalizer,
    trainer,
    evaluator,
):
    # Link all the steps artifacts together
    X_train, y_train, X_test, y_test = importer()
    X_trained_normed, X_test_normed = normalizer(X_train=X_train, X_test=X_test)
    model = trainer(X_train=X_trained_normed, y_train=y_train)
    evaluator(X_test=X_test_normed, y_test=y_test, model=model)


# Initialise the pipeline
p = mnist_pipeline(
    importer=importer_mnist(),
    normalizer=normalizer(),
    trainer=trainer(config=TrainerConfig(epochs=1)),
    evaluator=evaluator(),
)

# Run the pipeline
p.run()


# Define a new modified import data step to download the Fashion MNIST model
@step
def importer_fashion_mnist() -> Output(
    X_train=np.ndarray, y_train=np.ndarray, X_test=np.ndarray, y_test=np.ndarray
):
    """Download the MNIST data store it as an artifact"""
    (X_train, y_train), (
        X_test,
        y_test,
    ) = tf.keras.datasets.fashion_mnist.load_data()
    return X_train, y_train, X_test, y_test


# Load configuration options for the pipeline from a yaml file
mnist_config_file = os.path.join(os.path.dirname(__file__), "mnist_config.yaml")

# Initialise a new pipeline
fashion_p = mnist_pipeline(
    importer=importer_fashion_mnist(),
    normalizer=normalizer(),
    trainer=trainer(),
    evaluator=evaluator(),
).with_config(mnist_config_file)

# Run the new pipeline
fashion_p.run()
