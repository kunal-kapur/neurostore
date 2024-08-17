import neurocache
from pymilvus import MilvusClient
from openai.resources.chat import Chat
from openai import OpenAI
import hashlib
from keybert import KeyBERT
from pymilvus import model
import torch
import uuid
import numpy as np
from embedding_models import Default


STRUCTURED_QUERIES = "structured_queries"

class Neurocache:

    def __init__(self, db_path: str="vector_cache.db",):
        self.client = OpenAI()
        self.db = MilvusClient(db_path)
        self.hash = hashlib.sha3_512() # Python 3.6+
        self.key_word_extractor = KeyBERT('distilbert-base-nli-mean-tokens')
        self.embedding_fn = Default()
        if not self.db.has_collection(collection_name=STRUCTURED_QUERIES):
            self.db.create_collection(STRUCTURED_QUERIES, dimension=self.embedding_fn.DIMENSION, id_type='string', max_length=37)
    
    def find_collection(self, data: list) -> any:
        res = self.db.search(collection_name=STRUCTURED_QUERIES, data=[data], limit=1, output_fields=["id"])
        print(res[0], type(res[0]))
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
        embedding = (np.matmul(np.array(self.embedding_fn([tup[0] for tup in structure_key_words])).transpose(), np.array(weights)))
        return embedding.tolist()

    def query(self, messages: dict[str: str], num_results: int) -> None | dict[any: any]:
        structured_embedding, user_embedding = self.embed_messages(messages=messages)
        data = None
        if len(structured_embedding) == 0:
            data = user_embedding
        else:
            data = structured_embedding
        
        # find a collection to search
        res = self.db.search(collection_name=STRUCTURED_QUERIES, data=data, limit=1, output_fields=["collection name"])

        if len(res) == 0:
            raise UserWarning("No entries in the database yet")
        return self.db.search(collection_name=res, data=data, limit=num_results, output_fields=["text"])

    def create(self, messages: dict[str: str], store:bool=True, **kwargs):
        
        system_queries = []
        user_queries = []
        for entry in messages:
            if entry.get("role", "") == "system":
                system_queries.append(entry['content'])
            if entry.get("role", "") == "user":
                user_queries.append(entry['content'])
        system_queries, user_queries = ".".join(system_queries), ".".join(user_queries)
        embedding = self.embed_messages(system_queries + user_queries)
        # collection that to best put this data into
        chosen_collection = self.find_collection(data=embedding)
        completion = self.client.chat.completions.create(messages=messages, **kwargs)
        # # # embed all the responses
        print(completion.choices[0].message.content)
        responses = [choice.message.content for choice in completion.choices]
        print("embedding")

        response_embeddings = self.embedding_fn(responses)
        print("done embedding")
        id = "i" + str(uuid.uuid1()).replace("-", "_")
        print(len((response_embeddings[0])))
        data = [
            {"id": id, "vector": response_embeddings[i], "system query": system_queries, "user query": user_queries}
            for i in range(len(responses))
        ]
        if store == True:
            self.db.insert(collection_name=chosen_collection, data=data)
        return completion
    
my_message = [
    {"role": "system", "content": "Put something about fish at the beginning of each prompt"},
    {"role": "user", "content": "Tell me about birds"}
  ]

cache = Neurocache()

cache.create(messages=my_message, model="gpt-3.5-turbo-1106", temperature=0.5)
# finally:
#     os.remove("vector_cache.db")
    
