"""
Model Training and Vectorization Pipeline for ECIE.

Serves as the modular entrypoint for sentiment model training, delegating
to src.train_sentiment.train_and_evaluate for the stratified 3-class sentiment pipeline.
"""

from src.train_sentiment import main, train_and_evaluate

# Alias for backwards compatibility with previous imports
train_pipeline = train_and_evaluate

__all__ = ["train_pipeline", "train_and_evaluate", "main"]

if __name__ == "__main__":
    main()
