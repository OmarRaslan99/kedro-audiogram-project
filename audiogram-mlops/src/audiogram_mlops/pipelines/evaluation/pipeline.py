"""
This is a boilerplate pipeline 'evaluation'
generated using Kedro 1.2.0
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import evaluate_model

def create_pipeline(**kwargs) -> Pipeline:
    """
    Crée le pipeline d'évaluation pour valider la robustesse du modèle.
   
    """
    return pipeline([
        node(
            func=evaluate_model,
            # Ces entrées doivent exister dans ton catalog.yml
            inputs=["ml_model", "test_data", "test_labels"],
            # La sortie sera sauvegardée dans le dataset défini dans le catalogue
            outputs="performance_metrics",
            name="evaluate_model_node",
        ),
    ])