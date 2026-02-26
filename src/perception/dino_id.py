import numpy as np


class DinoIdentifier:
    def __init__(self,  reference_embeddings: dict):
        self.reference_embeddings = reference_embeddings

    def identify(self,cluster_points: np.ndarray, cluster_features: np.ndarray):
        """
        Return object_id, confidence
        """