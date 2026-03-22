# -----------------------------------------------------------------
#    PyPDFLoader : to load pdfs
#    CharacterTextSplitter : to split the documents into chunks
#
#    vector_db_lib : to store chunks in vectorized version
#
#    re : reqular expression, to get target content from a given string
#
#    shazri 2026
# -----------------------------------------------------------------
import numpy as np
import json

from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader

from langchain_text_splitters import CharacterTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter

import vector_db_lib
from vector_db_lib import add_to_vector_db

import re

import requests

import vector_db_lib
import llm_tools
import collate_and_summarize
import table_process

"""

    Basic configuration input from file, config.json

    'embedding_path' : folder location of the collection of vector to word
    'docs_path' : folder location of the collection of documents
    'vector_pickle_text' : pickle file that stores vector , text for text

"""

with open("config.json", "r") as f:
    config = json.load(f)

embedding_path = config["embedding_path"]
docs_path = config["docs_path"]
vector_db_all_pickle = config["vector_pickle_all"]

print("Embeddings:", embedding_path)
print("Docs folder:", docs_path)

"""
    To create a dictionary where,
    it has a map of word to vector.
    
"""

# create embedding index
embedding_index = {}
with open(embedding_path, encoding='utf-8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embedding_index[word] = coefs


"""
    docs which is the out is a list of dictionary
    each item in dictionary has

    each Document object has
    - metadata child object (dictionary)
    - page_content child object (string)

"""

docs = []

for pdf_file in Path(docs_path).rglob("*.pdf"):   # recursive search
    loader = PyPDFLoader(str(pdf_file))
    pages = loader.load()

    for page in pages:
        page.metadata["file"] = pdf_file.name
        page.metadata["path"] = str(pdf_file)

    docs.extend(pages)


"""
    The compilation of documents, docs is
    split into chunks.

"""


splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = splitter.split_documents(docs)

"""
    reformat to a list of dictionary
    each element represent a chunk

    - "text" : content, actual document content
    - "file" : filename
    - "page" : page number
    - "type" : type such as 'text' or 'image'
"""

records = []

for chunk in chunks:
    records.append({
        "text": chunk.page_content,
        "file": chunk.metadata["file"],
        "page": chunk.metadata["page"],
        "type": 'text'
    })


"""
    To load any saved vector database
"""

vector_db = vector_db_lib.load_vector_db(vector_db_all_pickle)


"""
    Place all items in a set
    set ensures all items are unique for comparison later
"""

existing_keys = set()

for item in vector_db:
    key = (item["file"], item["page"], item["text"] , item["type"])
    existing_keys.add(key)




len_vec = len(embedding_index[list(embedding_index.keys())[0]])




vector_db = add_to_vector_db(
    records,
    vector_db,
    existing_keys,
    embedding_index,
    len_vec,
    "text",
    "file",
    "page",
    "text",
    "image"
)


vector_db_lib.save_vector_db(vector_db,vector_db_all_pickle)


print("Total vectors in DB:", len(vector_db))

