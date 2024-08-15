import pathlib
import os

os.environ["OPENAI_API_KEY"] = pathlib.Path("key.txt").read_text()
os.environ["TOKENIZERS_PARALLELISM"] = "false"
print("HERE")
