"""
This is a boilerplate pipeline 'evaluation'
generated using Kedro 1.2.0
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import evaluate_model

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=evaluate_model,
            inputs=["ml_model", "X_test", "y_test"], 
            outputs="performance_metrics",
            name="evaluate_model_node",
        ),
    ])