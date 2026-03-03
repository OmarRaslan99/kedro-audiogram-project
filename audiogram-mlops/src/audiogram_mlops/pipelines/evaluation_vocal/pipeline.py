"""Pipeline definition for evaluation_vocal."""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import evaluate_vocal_model


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=evaluate_vocal_model,
                inputs=[
                    "ml_model_vocal",
                    "X_test_vocal",
                    "y_test_vocal",
                    "params:vocal_tolerances",
                ],
                outputs="performance_metrics_vocal",
                name="evaluate_vocal_model_node",
            ),
        ]
    )
