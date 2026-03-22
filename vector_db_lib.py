import pickle
import os

import json

from nltk.corpus import stopwords



def save_vector_db(vector_db,DB_PATH):
    with open(DB_PATH, "wb") as f:
        pickle.dump(vector_db, f)

def load_vector_db(DB_PATH):
    # If file does not exist, create empty DB
    if not os.path.exists(DB_PATH):
        vector_db = []
        save_vector_db(vector_db,DB_PATH)
        return vector_db

    # Try loading existing DB
    try:
        with open(DB_PATH, "rb") as f:
            vector_db = pickle.load(f)

        # ensure correct type
        if not isinstance(vector_db, list):
            raise ValueError("Invalid DB format")

        return vector_db

    except Exception as e:
        print("Vector DB corrupted or unreadable. Resetting DB:", e)
        vector_db = []
        save_vector_db(vector_db)
        return vector_db





def add_to_vector_db(
    records,
    vector_db,
    existing_keys,
    embedding_index,
    len_vec,
    text_,
    file_,
    page_,
    type_,
    image_
):
    """
    Adds new records to the vector database with embedding generation.

    This function:
    - Extracts text, file, page, and image fields from input records
    - Converts text into vector embeddings using a provided embedding index
    - Avoids duplicate entries using a unique key (file, page, text, type)
    - Appends only new records to the existing vector database
    - Filter in letters a to z and number 0 to 9.
    - Filter out stop words, word that they occur frequently but carry little unique meaning

    Parameters:
        records (list): List of input dictionaries (raw data)
        vector_db (list): Existing vector database (list of dicts)
        existing_keys (set): Set of unique keys to prevent duplicates
        embedding_index (dict): Word → vector mapping (e.g., GloVe)
        len_vec (int): Length of embedding vectors
        text_ (str): Key name for text field in records
        file_ (str): Key name for file path in records
        page_ (str): Key name for page number in records
        type_ (str): Type label for the record (e.g., "text", "image")
        image_ (str): Key name for image path in records

    Returns:
        list: Updated vector_db with newly added embedded records

    Notes:
        - Skips records with empty or missing text
        - Uses simple average of word embeddings
        - Safe for mixed schemas via configurable field names
    """
    
    import numpy as np
    import re

    stop_words = set(stopwords.words("english"))

    for r in records:

        text = r.get(text_, "")
        if not text:
            continue

        file = r.get(file_)
        page = r.get(page_, 0)
        r_type = type_
        r_image = r.get(image_)

        key = (file, page, text, r_type)

        if key in existing_keys:
            continue

        # Clean + tokenize
        sentence = re.sub(r'[^a-zA-Z0-9]', ' ', text)
        tokens = sentence.lower().split()

        tokens = [t for t in tokens if t not in stop_words]

        vector = np.zeros(len_vec)

        for token in tokens:
            if token in embedding_index:
                vector += embedding_index[token]

        if len(tokens) > 0:
            vector = vector / len(tokens)

        vector_db.append({
            "vector": vector,
            "text": text,
            "file": file,
            "page": page,
            "type": r_type,
            "image": r_image
        })

        existing_keys.add(key)

    return vector_db