# Neurostore

This package serves as a way to 'cache' previous queries from LLMs to reduce the number of API calls necessary by using Milvus vector databases. 
A user can use their choice of embedding model from OpenAI to small Albert models that can be run locally. This allows for more optimized storage and more efficient search when finding previous queries/answers


## Getting started 
```bash
pip install neurostore
```
### Configure your environment variables for you desired LLM 
(e.g 
```bash
  export OPENAI_API_KEY=....
```

#### Store an run as you would normally 
Query your desired LLM as you normally would
```python

cache = Neurostore()
my_message = [
    {
        "role": "system",
        "content": "Put something about fish at the beginning of each prompt",
    },
    {"role": "user", "content": "Tell me about birds"},
]
cache.create(messages=my_message, store=True, model="gpt-3.5-turbo-1106",temperature=0.5)
print(cache.query(messages=my_message, num_results=2))
```


## Diagram

![neurostore](https://github.com/user-attachments/assets/eb022955-508c-48eb-871a-2f9b789d9807)
