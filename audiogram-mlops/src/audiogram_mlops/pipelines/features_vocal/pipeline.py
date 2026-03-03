"""Pipeline definition for features_vocal."""

from kedro.pipeline import Pipeline, node

from .nodes import make_vocal_train_test


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=make_vocal_train_test,
                inputs="vocal_exams_clean",
                outputs=["X_train_vocal", "X_test_vocal",
                         "y_train_vocal", "y_test_vocal"],
                name="make_vocal_train_test_node",
            )
        ]
    )
