import json

import vector_db_lib
from vector_db_lib import add_to_vector_db



with open("config.json", "r") as f:
    config = json.load(f)

embedding_path       = config["embedding_path"]
docs_path            = config["docs_path"]
vector_db_all_pickle = config["vector_pickle_all"]
image_analysis_json  = config["image_analysis"]

print("Embeddings:", embedding_path)
print("Docs folder:", docs_path)



with open(image_analysis_json, "r", encoding="utf-8") as f:
    image_records = json.load(f)

print(len(image_records))
print(image_records[0])


# create embedding index
import numpy as np
embedding_index = {}
with open(embedding_path, encoding='utf-8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embedding_index[word] = coefs



vector_db = vector_db_lib.load_vector_db(vector_db_all_pickle)



existing_keys = set()

for item in vector_db:
    key = (item["file"], item["page"], item["text"] , item["type"])
    existing_keys.add(key)


import numpy as np
import re

len_vec = len(embedding_index[list(embedding_index.keys())[0]])




vector_db = add_to_vector_db(
    image_records,
    vector_db,
    existing_keys,
    embedding_index,
    len_vec,
    "description",
    "pdf",
    "page",
    "image",
    "image"
)


vector_db_lib.save_vector_db(vector_db,vector_db_all_pickle)


print("Total vectors in DB:", len(vector_db))

