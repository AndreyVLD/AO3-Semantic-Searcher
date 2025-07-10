import torch

from numpy import ndarray
from sentence_transformers import SentenceTransformer, CrossEncoder
from src.config import BI_ENCODER, MODEL_BATCH_SIZE, CROSS_ENCODER


class EmbeddingModel:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.bi_encoder = SentenceTransformer(BI_ENCODER)
        self.cross_encoder = CrossEncoder(CROSS_ENCODER)

    def embed_chunks(self, chunks: list[str]) -> ndarray:
        """
        Embed a list of text chunks using the SentenceTransformer Bi-Encoder model.
        """
        embeddings = self.bi_encoder.encode(chunks, device=self.device, batch_size=MODEL_BATCH_SIZE)
        return embeddings

    def cross_scores(self, pairs: list[tuple[str, str]]) -> ndarray:
        """
        Score a list of text pairs using the CrossEncoder model.
        """
        scores = self.cross_encoder.predict(pairs, batch_size=MODEL_BATCH_SIZE)
        return scores
