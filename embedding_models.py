from typing import Any, Literal, Mapping, Type, Union, List
from pymilvus import model
from abc import abstractmethod
from openai import OpenAI

EmbeddingCompletion = Literal["Default", "paraphrase-albert-small-v2", "text-embedding-3-small", "text-embedding-3-large"] 

class EmbeddingModel:
    @property
    @abstractmethod
    def dimension(self)->int:
        pass

    @property
    @abstractmethod
    def name(self)->str:
        pass

    @abstractmethod
    def __call__(self, words: Union[str, List[str]])->List[List[float]]:
        pass


class AlbertSmalllV2(EmbeddingModel):
    def __init__(self) -> None:
        self.embedding_model = model.DefaultEmbeddingFunction()

    @property
    def dimension(self)->int:
        return 768
    @property
    def name(self)->str:
        return "paraphrase-albert-small-v2"

    def __call__(self, words: Union[str, List[str]]) -> List[List[float]]:
        return self.embedding_model.encode_documents(words)
    
class OpenAI_Small(EmbeddingModel):
    def __init__(self) -> None:
        self.embedding_model = OpenAI()
    @property
    def dimension(self)->int:
        return 1536
    @property
    def name(self)->str:
        return "text-embedding-3-small"

    def __call__(self, words: Union[str, List[str]]) -> List[List[float]]:
        print(words)
        if isinstance(words, str):
            return [self.embedding_model.embeddings.create(input=words, model=self.name).data[0].embedding]

        return [result.embedding for result in self.embedding_model.embeddings.create(input=words, model=self.name).data]
    
class OpenAI_Large(EmbeddingModel):
    def __init__(self) -> None:
        self.embedding_model = OpenAI()
    @property
    def dimension(self)->int:
        return 3072
    @property
    def name(self)->str:
        return "text-embedding-3-large"

    def __call__(self, words: Union[str, List[str]]) -> List[List[float]]:
       return [result.embedding for result in self.embedding_model.embeddings.create(input=words, model=self.name).data]
    



EmbeddingCompletionMapping: Mapping[str, Type[EmbeddingModel]]= {
    "Default": AlbertSmalllV2,
    "paraphrase-albert-small-v2": AlbertSmalllV2,
    "text-embedding-3-small": OpenAI_Small,
    "text-embedding-3-large": OpenAI_Large
}
