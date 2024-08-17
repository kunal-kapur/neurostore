from typing import Any
from pymilvus import model

class Default:
    def __init__(self) -> None:
        self.DIMENSION= 768
        self.embedding_model =  model.DefaultEmbeddingFunction()
    def __call__(self, words: list[str]) -> Any:
        return self.embedding_model(words)