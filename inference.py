import neurostore
from pymilvus import MilvusClient
from openai import OpenAI
from keybert import KeyBERT
import uuid
import numpy as np
from embedding_models import Default
import os
from datetime import date
import json
from importlib.metadata import version
from openai.types.chat.chat_completion import ChatCompletion
from typing import Dict, List
import utils


STRUCTURED_QUERIES = "structured_queries"
NEURO_CACHE_PATH = "neurostore"

class Neurostore:

    def __init__(self, db_path: str="vector_cache.db", embedding_model_query="Default", embedding_model_answer="Default", api_key: str|None=None):

        if os.environ["OPENAI_API_KEY"] is not None and api_key is not None:
            api_key = api_key
        
        elif os.environ["OPENAI_API_KEY"] is None:
            raise ValueError("Please add an OpenAI key")

        self.client = OpenAI()
        self.embedding_fn = Default()

        # create folder for db and info
        if not os.path.exists(path=NEURO_CACHE_PATH):
            os.mkdir(NEURO_CACHE_PATH)
        if not db_path.endswith(".db"):
            db_path += ".db"
        new_db_path = os.path.join(NEURO_CACHE_PATH, db_path)

        # create json info for the database
        if not os.path.exists(new_db_path):
            info = {"neurostore version": version("neurostore"), 
                "creation date": date.today().strftime("%m-%d-%Y"),
                "embedding model": self.embedding_fn.NAME,
                "embedding dimension": self.embedding_fn.DIMENSION}
            with open(f"{os.path.splitext(new_db_path)[0]}_info.json", 'w') as f:
                json.dump(info, f)
        self.db = MilvusClient(new_db_path)
            
        self.key_word_extractor = KeyBERT('distilbert-base-nli-mean-tokens')

        if not self.db.has_collection(collection_name=STRUCTURED_QUERIES):
            self.db.create_collection(STRUCTURED_QUERIES, dimension=self.embedding_fn.DIMENSION, id_type='string', max_length=37)
    
    def find_collection(self, data: list) -> any:
        res = self.db.search(collection_name=STRUCTURED_QUERIES, data=[data], limit=1, output_fields=["id"])
        # no collections exist or nothing related to the query
        if len(res[0]) == 0 or abs(res[0][0]['distance']) < 0.2:
            unique_id = str(uuid.uuid1())
            unique_id = "i" + unique_id.replace("-", "_")
            self.db.create_collection(unique_id, dimension=self.embedding_fn.DIMENSION, id_type="string", max_length=37)
            self.db.insert(collection_name=STRUCTURED_QUERIES, data=[{"id": unique_id, "vector": data}])
            print("created", unique_id)
            return unique_id
        
        return res[0][0]['entity']['id']
    
    def embed_messages(self, queries: str)->np.array:
        structure_key_words, weights = zip(*self.key_word_extractor.extract_keywords(queries))
        embedding = (np.matmul(np.array(self.embedding_fn([tup[0] for tup in structure_key_words]), dtype='float64').transpose(), np.array(weights, dtype='float64')))
        return embedding.tolist()

    def query(self, messages: List[Dict[str, str]], num_results: int) -> List[List[dict]]:
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
        embedding = self.embed_messages(".".join(system_queries) + ".".join(user_queries))
        # find a collection to search
        res = self.db.search(collection_name=STRUCTURED_QUERIES, data=[embedding], limit=1, output_fields=["id"])[0]

        if len(res) == 0:
            raise UserWarning("No entries in the database yet")
            return []
        return self.db.search(collection_name=res[0]['id'], data=[embedding], limit=num_results, output_fields=["system_query", "user_query", "response"])

    def create(self, messages: Dict[str, str], store:bool=True, **kwargs)->ChatCompletion:
        """Create an API call to the OpenAI API and optionally store it

        Args:
            messages (Dict[str, str]): messages passed to the openAI API
            store (bool, optional): If the given query and response should be cached. Defaults to True.

        Returns:
            ChatCompletion: the chat completion the openAI API would return
        """
        system_queries, user_queries = utils.combine_queries(messages=messages)
        embedding = self.embed_messages(".".join(system_queries) + ".".join(user_queries))
        # collection that to best put this data into
        chosen_collection = self.find_collection(data=embedding)
        completion = self.client.chat.completions.create(messages=messages, **kwargs)
        # embed all the responses
        responses = [choice.message.content for choice in completion.choices]
        response_embeddings = self.embedding_fn(responses)
        id = "i" + str(uuid.uuid1()).replace("-", "_")

        data = [
            {"id": id, "vector": response_embeddings[i], "system_query": system_queries, "user_query": user_queries, "response": responses[i]}
            for i in range(len(responses))
        ]
        if store == True:
            self.db.insert(collection_name=chosen_collection, data=data)
        return completion
    
my_message = [
    {"role": "system", "content": "Put something about fish at the beginning of each prompt"},
    {"role": "user", "content": "Tell me about birds"}
  ]

cache = Neurostore()

# client = OpenAI()
# client.chat.completions.create(model="")

cache.create(messages=my_message, model="gpt-3.5-turbo-1106", temperature=0.5)
print(cache.query(messages=my_message, num_results=1))
# finally:
#     os.remove("vector_cache.db")
    
