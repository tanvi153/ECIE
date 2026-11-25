"""
Machine Learning Models and Inference for ECIE.
"""

from src.models.predict import predict_sentiment
from src.models.train import train_pipeline

__all__ = ["train_pipeline", "predict_sentiment"]
