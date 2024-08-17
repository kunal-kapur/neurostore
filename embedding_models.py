from typing import Any, Literal, Mapping
from pymilvus import model
from abc import abstractmethod



EmbeddingCompletion = Literal["Default", "paraphrase-albert-small-v2"] 

class EmbeddingModel:
    @property
    @abstractmethod
    def dimension(self):
        pass

    @property
    @abstractmethod
    def name(self):
        pass


class AlbertSmalllV2(EmbeddingModel):
    def __init__(self) -> None:
        self.embedding_model = model.DefaultEmbeddingFunction()

    @property
    def dimension(self):
        return 768
    @property
    def name(self):
        return "paraphrase-albert-small-v2"

    def __call__(self, words: list[str]) -> Any:
        return self.embedding_model(words)
    



EmbeddingCompletionMapping: Mapping[str, EmbeddingModel] = {
    "Default": AlbertSmalllV2,
    "paraphrase-albert-small-v2": AlbertSmalllV2
    
}
