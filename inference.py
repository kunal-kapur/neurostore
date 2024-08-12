import neurocache
from pymilvus import MilvusClient
from openai.resources.chat import Chat
from openai import OpenAI
import hashlib
from keybert import KeyBERT
from pymilvus import model
import torch
import uuid
import embedding_models


STRUCTURED_QUERIES = "structured queries"

class Neurocache:

    def __init__(self, db_path: str="vector_cache.db",):
        self.client = OpenAI()
        self.db = MilvusClient(db_path)
        self.hash = hashlib.sha3_512() # Python 3.6+
        self.key_word_extractor = KeyBERT('distilbert-base-nli-mean-tokens')
        if not client.has_collection(collection_name=STRUCTURED_QUERIES):
            self.db.create_collection(STRUCTURED_QUERIES, dimension=768, id_type='str')
        self.embedding_fn = embedding_models.Default()
    
    def find_collection(self, data: torch.tensor) -> any:
        res = self.db.search(collection_name=STRUCTURED_QUERIES, data=data, limit=1, output_fields=["collection name"])
        if len(res) == 0 or abs(res[0]['distance']) < 0.2:
            unique_id = uuid.uuid1()
            self.db.create_collection(collection_name=unique_id, dimension=self.embedding_fn.DIMENSION, id_type="str")
            self.db.insert(collection_name=STRUCTURED_QUERIES, data={"collection name": unique_id})
            return unique_id
        
        return res[0]['entity']['collection name']
    
    def embed_messages(self, messages: dict[str: str]):
        system_queries = []
        user_queries = []
        for entry in messages:
            if entry.get("role", "") == "system":
                system_queries.append(entry['content'])
            if entry.get("role", "") == "user":
                user_queries.append(entry['content'])
        structure_key_words, weights = zip(*self.key_word_extractor(".".join(system_queries)))
        structured_embedding = torch.sum(torch.dot(self.embedding_fn([tup[0] for tup in structure_key_words]), weights))

        user_embedding = self.embedding_fn(user_queries)

        return structured_embedding, user_embedding

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
        structured_embedding, user_embedding = self.embed_messages(messages=messages)
        # collection that to best put this data into
        chosen_collection = self.find_collection(embedding=structured_embedding)
        completion = self.client.chat.completions.create(kwargs, messages=messages)
        # embed all the responses
        responses = [choice.message.content for choice in completion]
        embeddings = [self.embedding_fn(response) for response in responses]
        data = [
            {"id": uuid.uuid1(), "vector": embeddings[i], "text": responses[i], }
            for i in range(len(responses))
        ]
        if store == True:
            self.db.insert(collection_name=chosen_collection, data=data)
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