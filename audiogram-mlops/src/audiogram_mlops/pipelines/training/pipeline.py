"""
This is a boilerplate pipeline 'training'
generated using Kedro 1.2.0
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import train_model

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=train_model,
                # Attention : Doit correspondre exactement au catalogue
                inputs=["X_train", "y_train"],
                outputs="ml_model",
                name="train_model_node",
            ),
        ]
    )