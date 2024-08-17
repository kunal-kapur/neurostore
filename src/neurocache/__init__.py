import pathlib
import os 


os.environ["OPENAI_API_KEY"] = pathlib.Path("key.txt").read_text()
print("HERE")