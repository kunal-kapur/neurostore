import neurocache
from pymilvus import MilvusClient
from openai.resources.chat import Chat
from openai import OpenAI



class Neurocache:
    def __init__(self, name: str="vector_cache.db"):
        self.client = OpenAI()
        self.db = MilvusClient(name)
    
    def create(self, messages: dict[str: str], use_db:bool=False, store:bool=True, **kwargs):
        if use_db is True and store is True:
            raise ValueError("use_db and store can't both be True")

        system_queries = []
        user_queries = []
        for entry in messages:
            if entry.get("role", "") == "system":
                system_queries.append(entry['content'])
            if entry.get("role", "") == "user":
                system_queries.append(entry['content'])
        completion = self.client.chat.completions.create(kwargs, messages=messages)

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