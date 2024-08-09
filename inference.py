import neurocache
from pymilvus import MilvusClient
from openai.resources.chat import Chat
from openai import OpenAI
import hashlib
from keybert import KeyBERT
from pymilvus import model
import torch
import uuid


STRUCTURED_QUERIES = "structured queries"

class Neurocache:
    def __init__(self, db_path: str="vector_cache.db",):
        self.client = OpenAI()
        self.db = MilvusClient(db_path)
        self.hash = hashlib.sha3_512() # Python 3.6+
        self.key_word_extractor = KeyBERT('distilbert-base-nli-mean-tokens')
        if not client.has_collection(collection_name=STRUCTURED_QUERIES):
            self.db.create_collection(STRUCTURED_QUERIES, dimension=768, id_type='str')
        self.embedding_fn = model.DefaultEmbeddingFunction()
    
    def find_collection(self, embedding: torch.tensor) -> any:
        res = self.db.search(collection_name=STRUCTURED_QUERIES, limit=1, output_fields=["collection name"])

        if len(res) == 0 or abs(res[0]['distance']) < 0.2:
            unique_id = uuid.uuid1()
            self.db.create_collection(collection_name=unique_id, dimension=768, id_type="str")
            self.db.insert(collection_name=STRUCTURED_QUERIES, data={"collection name": unique_id})
            return unique_id
        return res[0]['entity']['collection name']
    

    def create(self, messages: dict[str: str], use_db:bool=False, store:bool=True, **kwargs):
        if use_db is True and store is True:
            raise ValueError("use_db and store can't both be True")

        system_queries = []
        user_queries = []
        for entry in messages:
            if entry.get("role", "") == "system":
                system_queries.append(entry['content'])
            if entry.get("role", "") == "user":
                user_queries.append(entry['content'])

        key_words, weights = zip(*self.key_word_extractor(".".join(system_queries)))

        # weight average operation to get single vector to represent everything
        structured_embedding = torch.sum(torch.dot(self.embedding_fn([tup[0] for tup in key_words]), weights))

        completion = self.client.chat.completions.create(kwargs, messages=messages)
        responses = [choice.message.content for choice in completion]

        
        # self.db.create_collection()
        self.hash.update

        return completion


client = OpenAI()
completion = client.chat.completions.create(
  model="gpt-3.5-turbo-1106",
  messages=[
    {"role": "system", "content": "Put something about fish at the beginning of each prompt"},
    {"role": "system", "content": "Finish every answer with something about pandas."},
  ],
  temperature=0.5
)

print(completion.choices[0])

# with open('out.txt', 'a') as f:
#     f.write((completion.choices[0].message.content) + "\n")

# completion = client.chat.completions.create(
#   model="gpt-3.5-turbo-1106",
#   messages=[
#     {"role": "user", "content": "What did I say in my last API call?."}
#   ],
#   max_tokens=40,
#   temperature=0.5
# )

# with open('out.txt', 'a') as f:
#     f.write((completion.choices[0].message.content) + "\n")