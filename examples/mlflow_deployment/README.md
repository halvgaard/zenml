# ZenML model serving with MLflow deployments

The popular open source MLflow platform is known primarily for its great
[experiment tracking and visualization](https://mlflow.org/docs/latest/tracking.html)
user experience. Among its many features, MLflow also provides a standard format
for packaging ML models and deploying them for real-time serving using a range
of deployment tools.

This example continues the story around the ZenML integration for MLflow experiment
tracking showcased in the [mlflow_tracking example](../mlflow_tracking) and adds
deploying MLflow models locally with its
[local built-in deployment server](https://mlflow.org/docs/latest/models.html#deploy-mlflow-models).

## Overview

This example extends the ZenML pipeline discussed in the
[mlflow_tracking example](../mlflow_tracking) with model serving capabilities.

The pipeline uses the
[Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) dataset and
trains a classifier using [Tensorflow (Keras)](https://www.tensorflow.org/) using different
hyperparameter values (epochs and learning rate) that can also be supplied as command line
arguments.

In the first half of the pipeline, ZenML's MLflow tracking integration is used to log
the hyperparameter values as well as the trained model itself and the model evaluation metrics
as MLflow experiment tracking artifacts into the local mlflow backend.

The second half of the pipeline is concerned with implementing a continuous model
deployment workflow: a local MLflow deployment server is launched to serve the latest MLflow
model. The deployment server is running locally as a daemon process and will survive
the execution of the example script.

The pipeline has caching enabled to avoid re-training the model if the hyperparameter values
don't change. In the post-execution phase, the example interacts with the MLflow deployment
server to perform a simple prediction using the builtin client.

This example uses an mlflow setup that is based on the local filesystem as
orchestrator and artifact store. See the [mlflow
documentation](https://www.mlflow.org/docs/latest/tracking.html#scenario-1-mlflow-on-localhost)
for details.

## Run it locally

### Pre-requisites
In order to run this example, you need to install and initialize ZenML:

```shell
# install CLI
pip install zenml

# install ZenML integrations
zenml integration install mlflow
zenml integration install tensorflow

# pull example
zenml example pull mlflow_deployment
cd zenml_examples/mlflow_deployment

# initialize
zenml init
```

### Run the project
To run the pipeline locally:

```shell
python run.py
```

Re-running the example with different hyperparameter values will re-train
the model and restart the MLflow deployment server to serve the new model:

```shell
python run.py --epochs=10 --learning_rate=0.1
```

If the input argument values are not changed, the pipeline caching feature
will kick in and the model will not be re-trained. Independently of the pipeline
run, the currently running MLflow deployment server will be used to perform an
example prediction in the post-execution phase.

Finally, to stop the prediction server, simply pass the `--stop-service` flag
to the example script:

```shell
python run.py --stop-service
```

### Clean up

To stop the prediction server running in the background, pass the
`--stop-service` flag to the example script:

```shell
python run.py --stop-service
```

Then delete the remaining ZenML references.

```shell
rm -rf zenml_examples
```

## SuperQuick `mlflow` run

If you're really in a hurry and you want just to see this example pipeline run,
without wanting to fiddle around with all the individual installation and
configuration steps, just run the following:

```shell
zenml example run mlflow_deployment
```
