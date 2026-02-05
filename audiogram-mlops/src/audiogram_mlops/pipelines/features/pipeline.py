from kedro.pipeline import Pipeline, node
from .nodes import make_train_test

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline([
        node(
            func=make_train_test,
            inputs="tonal_exams_clean",
            outputs=["X_train", "X_test", "y_train", "y_test"],
            name="make_train_test_node",
        )
    ])
