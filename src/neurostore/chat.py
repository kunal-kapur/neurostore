import neurostore
from pymilvus import MilvusClient  # type: ignore
from openai import OpenAI
from keybert import KeyBERT  # type: ignore
import uuid
import numpy as np
from neurostore.embedding_models import EmbeddingCompletion, EmbeddingCompletionMapping
import os
from datetime import date
import json
from importlib.metadata import version
from openai.types.chat.chat_completion import ChatCompletion
from typing import Dict, List, Union, Any
import neurostore.utils as utils
import logging


STRUCTURED_QUERIES = "structured_queries"
NEURO_STORE_PATH = "neurostore"


class Neurostore:

    def __init__(
        self,
        db_path: str = "./vector_cache.db",
        embedding_model_query: EmbeddingCompletion = "Default",
        embedding_model_answer: EmbeddingCompletion = "Default",
        api_key: Union[str, None] = None,
    ):
        """_summary_

        Args:
            db_path (str, optional): Path for where the vector database should be stored. Defaults to "./vector_cache.db".
            embedding_model_query (EmbeddingCompletion, optional): Type of embedding model for queries to choose from. Defaults to "Default".
            embedding_model_answer (EmbeddingCompletion, optional): Type of embedding model for answer to choose from. Defaults to "Default".
            api_key (Union[str, None], optional): The users API key. Defaults to None.

        Raises:
            ValueError: If there is no API key that is found in environment variables or configuration file
            UserWarning: If the original data file doesn't exist for the database
        """
        self.client = OpenAI()

        config_info = utils.config_import()

        if os.environ["OPENAI_API_KEY"] is None and api_key is not None:
            api_key = api_key
        elif (
            os.environ["OPENAI_API_KEY"] is not None
            and utils.safe_get(config_info, None, "api_keys", "OpenAI") is not None
        ):
            api_key = utils.safe_get(config_info, None, "api_keys", "OpenAI")
        elif os.environ["OPENAI_API_KEY"] is None:
            raise ValueError("Please add an OpenAI key")

        db_path = utils.safe_get(config_info, db_path, "database_path").strip()
        head, tail = os.path.split(db_path)
        # create folder for db and info
        if not os.path.exists(path=head) and len(head) > 0:
            os.makedirs(head)
        if not tail.endswith(".db"):
            tail += ".db"
        new_db_path = os.path.join(head, tail)

        info_path = f"{os.path.splitext(new_db_path)[0]}_info.json"
        # create json info for the database
        if not os.path.exists(new_db_path):
            self.embedding_model_query = EmbeddingCompletionMapping[
                str(
                    utils.safe_get(
                        config_info, embedding_model_query, "embedding_models", "query"
                    )
                )
            ]()
            self.embedding_model_answer = EmbeddingCompletionMapping[
                str(
                    utils.safe_get(
                        config_info,
                        embedding_model_answer,
                        "embedding_models",
                        "answer",
                    )
                )
            ]()
            info = {
                "neurostore version": version("neurostore"),
                "creation date": date.today().strftime("%m-%d-%Y"),
                "embedding model query": self.embedding_model_query.name,
                "embedding model query dimension": self.embedding_model_query.dimension,
                "embedding model answer": self.embedding_model_answer.name,
                "embedding model answer dimension": self.embedding_model_answer.dimension,
            }
            with open(info_path, "w") as f:
                json.dump(info, f)
        else:
            logging.log(level=20, msg="Using existing configuration")

            if not os.path.exists(info_path):
                raise UserWarning(
                    f"{info_path} doesn't exist; please write in database"
                )
            with open(info_path, "r") as f:
                set_config = json.load(f)
            self.embedding_model_query = EmbeddingCompletionMapping[
                str(
                    utils.safe_get(
                        set_config, embedding_model_query, "embedding model query"
                    )
                )
            ]()
            self.embedding_model_answer = EmbeddingCompletionMapping[
                str(
                    utils.safe_get(
                        set_config, embedding_model_answer, "embedding model answer"
                    )
                )
            ]()

        self.db = MilvusClient(new_db_path)
        self.db_path = new_db_path

        self.key_word_extractor = KeyBERT("distilbert-base-nli-mean-tokens")

        if not self.db.has_collection(collection_name=STRUCTURED_QUERIES):
            self.db.create_collection(
                STRUCTURED_QUERIES,
                dimension=self.embedding_model_query.dimension,
                id_type="string",
                max_length=37,
            )

    def find_collection(self, data: list) -> Any:
        res = self.db.search(
            collection_name=STRUCTURED_QUERIES,
            data=[data],
            limit=1,
            output_fields=["id"],
        )
        # no collections exist or nothing related to the query
        if len(res[0]) == 0 or abs(res[0][0]["distance"]) < 0.2:
            unique_id = str(uuid.uuid1())
            unique_id = "i" + unique_id.replace("-", "_")
            self.db.create_collection(
                unique_id,
                dimension=self.embedding_model_answer.dimension,
                id_type="string",
                max_length=37,
            )
            self.db.insert(
                collection_name=STRUCTURED_QUERIES,
                data=[{"id": unique_id, "vector": data}],
            )
            return unique_id

        return res[0][0]["entity"]["id"]

    def embed_messages(self, queries: str) -> list[int]:
        structure_key_words, weights = zip(
            *self.key_word_extractor.extract_keywords(queries)
        )

        structured_words_embedding = self.embedding_model_query(
            list(structure_key_words)
        )

        embedding = np.matmul(
            np.array(
                structured_words_embedding,
                dtype="float64",
            ).transpose(),
            np.array(weights, dtype="float64"),
        )
        return embedding.tolist()

    def query(
        self, messages: List[Dict[str, str]], num_results: int = 1
    ) -> List[List[dict]]:
        """Create an API call to the OpenAI API and optionally store it.

        Args:
            messages (List[Dict[str, str]]): messages passed to the openAI API
            num_results (int): number of results that should be given

        Raises:
            UserWarning: If there are no entries in the database

        Returns:
            List[Dict[str, any]]: returned value from querying the Milvus database
        """

        system_queries, user_queries = utils.combine_queries(messages=messages)

        joint_message = ".".join(system_queries) + ".".join(user_queries)
        embedding = self.embedding_model_query(joint_message)
        # find a collection to search
        res = self.db.search(
            collection_name=STRUCTURED_QUERIES,
            data=embedding,
            limit=1,
            output_fields=["id"],
        )[0]

        if len(res) == 0:
            raise UserWarning("No entries in the database yet")

        answer_embedding = embedding
        if self.embedding_model_answer.name != self.embedding_model_query.name:
            answer_embedding = self.embedding_model_answer(joint_message)

        return self.db.search(
            collection_name=res[0]["id"],
            data=answer_embedding,
            limit=num_results,
            output_fields=["system_query", "user_query", "response"],
        )

    def create(
        self, messages: Any, store: bool = True, model="gpt-3.5-turbo-1106", **kwargs
    ) -> ChatCompletion:
        """Create an API call to the OpenAI API and optionally store it

        Args:
            messages (Dict[str, str]): messages passed to the openAI API
            store (bool, optional): If the given query and response should be cached. Defaults to True.

        Returns:
            ChatCompletion: the chat completion the openAI API would return
        """
        system_queries, user_queries = utils.combine_queries(messages=messages)
        embedding = self.embed_messages(
            ".".join(system_queries) + ".".join(user_queries)
        )
        # collection that to best put this data into
        chosen_collection = self.find_collection(data=embedding)
        completion = self.client.chat.completions.create(
            model=model, messages=messages, **kwargs
        )
        # embed all the responses
        responses = [choice.message.content for choice in completion.choices]
        response_embeddings = self.embedding_model_answer(responses)
        id = "i" + str(uuid.uuid1()).replace("-", "_")

        data = [
            {
                "id": id,
                "vector": response_embeddings[i],
                "system_query": system_queries,
                "user_query": user_queries,
                "response": responses[i],
            }
            for i in range(len(responses))
        ]
        if store == True:
            self.db.insert(collection_name=chosen_collection, data=data)
        return completion
