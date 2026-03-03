"""Pipeline definition for training_vocal."""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import train_vocal_model


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=train_vocal_model,
                inputs=["X_train_vocal", "y_train_vocal", "params:model_options_vocal"],
                outputs="ml_model_vocal",
                name="train_vocal_model_node",
            ),
        ]
    )
