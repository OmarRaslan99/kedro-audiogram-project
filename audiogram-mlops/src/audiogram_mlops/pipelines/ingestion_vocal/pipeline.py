"""Pipeline definition for ingestion_vocal."""

from kedro.pipeline import Pipeline, node

from .nodes import clean_vocal_exams


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=clean_vocal_exams,
                inputs="vocal_exams_raw",
                outputs="vocal_exams_clean",
                name="clean_vocal_exams_node",
            )
        ]
    )
