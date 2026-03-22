#-----------------------------------------------------------
# Toola and Interfaces that being used to interact with Ollama
# shazri 2026
#-----------------------------------------------------------

import requests

import vector_db_lib
import numpy as np
import json

import re
# -----------------------------
# Load configuration
# -----------------------------
with open("config.json", "r") as f:
    config = json.load(f)

embedding_path = config["embedding_path"]
docs_path = config["docs_path"]
vector_db_all_pickle = config["vector_pickle_all"]




# -----------------------------
# multi use base ask the ollama
# -----------------------------


def ask_ollama(prompt):
    r = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "llama3",
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0,     
                "top_p": 1,
                "top_k": 1
            }
        }
    )
    return r.json()["response"]

# -------------------------------
# this prompt will be used if the in 'REQUIREMENTS' is detected
# ------------------------------

def ask_requirements(context):

    prompt =  f"""
You are an engineering technical analyst analyzing tender documents.

IMPORTANT RULES:
- DO NOT summarize
- Process each SOURCE independently
- Copy file and page EXACTLY from SOURCE
- Do NOT guess metadata
- Do NOT skip any field (use "None" if missing)
- Extract exact phrases even if incomplete

Allowed categories ONLY:
- TECHNICAL_SPECS
- TIMELINES
- DOCUMENTATION
- QUALITY_STANDARDS

Allowed classification ONLY:
- MANDATORY
- OPTIONAL

Allowed compliance ONLY:
- YES
- NO
- PARTIAL 
- UNKNOWN

context to infer compliance:
From a point of view of a mid-sized electrical infrastructure company bidding for a high-voltage transmission project worth approximately 100-150 Million.

Return format EXACTLY like this example:

Requirement: The contractor shall provide safety helmets.
Category of Requirement: QUALITY_STANDARDS
File: sample.pdf
Page: 12
Type: text
Image: None
Classification: Mandatory
Compliance: Yes

Now follow this format strictly.

SOURCES:
{context}
"""
    r = ask_ollama(prompt)



    return r





# -----------------------------
# Load embeddings
# -----------------------------
embedding_index = {}
with open(embedding_path, encoding='utf-8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embedding_index[word] = coefs

len_vec = len(next(iter(embedding_index.values())))

# -----------------------------
# Load vector database
# -----------------------------
vector_db = vector_db_lib.load_vector_db(vector_db_all_pickle)

# -----------------------------
# Cosine similarity / search
# -----------------------------
def cosine(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def search(query_vector, k=5):

    print('in search')
    print(query_vector)

    scores = []

    for item in vector_db:
        sim = cosine(query_vector, item["vector"])
        scores.append((sim, item))

    scores.sort(key=lambda x: x[0], reverse=True)

    return [x[1] for x in scores[:k]]

def vectorize(text):
    text = re.sub(r'[^a-zA-Z0-9]', ' ', text)
    tokens = text.lower().split()
    vec = np.zeros(len_vec)
    for token in tokens:
        if token in embedding_index:
            vec += embedding_index[token]
    if len(tokens) > 0:
        vec = vec / len(tokens)
    return vec