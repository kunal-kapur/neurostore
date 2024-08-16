from typing import Any, Literal, Mapping
from pymilvus import model
from abc import abstractmethod
from openai import OpenAI



EmbeddingCompletion = Literal["Default", "paraphrase-albert-small-v2", "text-embedding-3-small"] 

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
    
class OpenAI_Small(EmbeddingModel):
    def __init__(self) -> None:
        self.embedding_model = OpenAI()
    @property
    def dimension(self):
        return 1536
    @property
    def name(self):
        return "text-embedding-3-small"

    def __call__(self, words: list[str]) -> Any:
        return self.embedding_model.embeddings.create(input=words, model=self.name)
    
class OpenAI_Large(EmbeddingModel):
    def __init__(self) -> None:
        self.embedding_model = OpenAI()
    @property
    def dimension(self):
        return 3072
    @property
    def name(self):
        return "text-embedding-3-large"

    def __call__(self, words: list[str]) -> Any:
        return self.embedding_model.embeddings.create(input=words, model=self.name)
    



EmbeddingCompletionMapping: Mapping[str, EmbeddingModel] = {
    "Default": AlbertSmalllV2,
    "paraphrase-albert-small-v2": AlbertSmalllV2,
    "text-embedding-3-small": OpenAI_Small,
    "text-embedding-3-large": OpenAI_Large
}
