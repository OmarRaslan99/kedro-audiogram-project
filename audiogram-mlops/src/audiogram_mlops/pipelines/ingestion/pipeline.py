from kedro.pipeline import Pipeline, node
from .nodes import clean_tonal_exams

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline([
        node(
            func=clean_tonal_exams,
            inputs="tonal_exams_raw",
            outputs="tonal_exams_clean",
            name="clean_tonal_exams_node",
        )
    ])
