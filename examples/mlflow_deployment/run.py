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

from pipeline import (
    TrainerConfig,
    mlflow_example_pipeline,
    importer_mnist,
    normalizer,
    tf_trainer,
    tf_evaluator,
    predictor,
    batch_inference,
    inference_evaluator,
)

if __name__ == "__main__":

    # Initialize a pipeline run
    run_1 = mlflow_example_pipeline(
        importer=importer_mnist(),
        normalizer=normalizer(),
        trainer=tf_trainer(config=TrainerConfig(epochs=5, lr=0.0003)),
        evaluator=tf_evaluator(),
        predictor=predictor(),
        batch_inference=batch_inference(),
        inference_evaluator=inference_evaluator(),
    )

    run_1.run()
