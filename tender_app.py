#-------------------------------------------------------------------------------
# Streamlit app, infer intent 1) REQUIREMENTS 2) BOQ 3) GENERAL
# infer what is a requirement using OLLAMA
# extract from source using sentence transformer
# shazri 2026
#------------------------------------------------------------------------------
import streamlit as st
import numpy as np
import pandas as pd
import json
import re
import joblib

import vector_db_lib
import llm_tools
import collate_and_summarize
import table_process

from sentence_transformers import SentenceTransformer, util

# Load the model only once
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# -----------------------------
# Load configuration
# -----------------------------
with open("config.json", "r") as f:
    config = json.load(f)

embedding_path = config["embedding_path"]
docs_path = config["docs_path"]
vector_db_all_pickle = config["vector_pickle_all"]



# -----------------------------
# Intent prediction
# -----------------------------
clf = joblib.load("intent_clf.pkl")
vectorizer = joblib.load("intent_vectorizer.pkl")

def predict_intent(query):
    vec = vectorizer.transform([query])
    return clf.predict(vec)[0]

# -----------------------------
# Streamlit config
# -----------------------------
st.set_page_config(page_title="Test App", layout="wide")
st.title("mvp tender intelligence")

# -----------------------------
# Chat history
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# -----------------------------
# Session state for tables and download buttons
# -----------------------------
for key in ["c_df", "c_df_text","c_df_image", "summarize_c_df", "dframefinal" , "download1_clicked" , "download1_1_clicked", "download1_2_clicked", "download2_clicked" , "download3_clicked" ]:
    if key not in st.session_state:
        st.session_state[key] = None if "df" in key else False

# -----------------------------
# Cached CSV generator
# -----------------------------
@st.cache_data
def to_csv_bytes(df):
    return df.to_csv(index=False).encode("utf-8")

# -----------------------------
# User input
# -----------------------------
query = st.chat_input("Ask something...")

# 1️Generate tables
if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    query_ = query + " timeline deadline completion period specification material requirement document quality standard condition"
    intent = str(predict_intent(query))
    q_vec = llm_tools.vectorize(query_)

    intent_ = 'intent detected: ' + intent
    intent_

    # -----------------------------
    # blocks
    # -----------------------------
    if intent == 'GENERAL':
        results = llm_tools.search(q_vec, k=5)
        context = ""
    
        for r in results:
            context += f"""
        SOURCE:
        file: -->{r['file']}
        page: -->{r['page']}
        
        content:
        {r['text']}
        
        ====
        """
        context = query + ' ' + context
    
        answer = llm_tools.ask_ollama(context)

        answer = answer.replace("\n", "  \n")

        with st.chat_message("assistant"):
            st.markdown(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})


    
    if intent == 'BOQ':
        st.session_state.dframefinal = table_process.process_table()
        
    if intent == "REQUIREMENTS":
        results = llm_tools.search(q_vec, k=55)
        collate_answer = ""

        for r in results:
            context = f"""
SOURCE:
file: {r['file']}
page: {r['page']}
type: {r.get('type','text')}
"""
            if r.get("image"):
                context += f"image: {r['image']}\n"

            context += f"""
content:
{r['text']}

====
"""
            answer = llm_tools.ask_requirements(context)

            # Encode sentences
            source_text       = r['text']
            source_sentences  = source_text.split(". ")
            temp_mat = re.search(r"Requirement:\s*(.*)", answer)
            if temp_mat:
                excerpt = temp_mat.group(1) + ". " + query_
            else:
                excerpt = query_
                
            source_embeddings = model.encode(source_sentences, convert_to_tensor=True)
            excerpt_embedding = model.encode(excerpt, convert_to_tensor=True)
    
            # Find best match
            similarities = util.cos_sim(excerpt_embedding, source_embeddings)
            best_idx = similarities.argmax()
            best_match = source_sentences[best_idx]
    
            # Append to the Requirement line only
            # Only append Excerpt after Requirement, not anywhere else
            answer = re.sub(
                r"^(Requirement:.*?)(?=\nCategory of Requirement:)",  # non-greedy up to Category
                lambda m: m.group(1) + "  \nExcerpt from Source: " + best_match,
                answer,
                flags=re.DOTALL | re.MULTILINE
            )
            
            
            answer = answer.replace("\n", "  \n")

            # sometimes llms hallucinate
            answer = re.sub(
                r"Type:\s*.*",
                f"Type: {r.get('type','text')}",
                answer
            )
            
            answer = re.sub(
                r"File:\s*.*",
                lambda m: f"File: {r['file']}",
                answer
            )
            
            # Fix Page
            answer = re.sub(
                r"Page:\s*.*",
                f"Page: {r['page']}",
                answer
            )

            with st.chat_message("assistant"):
                st.markdown(answer)

            st.session_state.messages.append({"role": "assistant", "content": answer})
            # Compile answers
            collate_answer = collate_answer + '\n' + answer
    
            # Clean all LLM fluff before parsing
            collate_answer = re.sub(
                    r"(?i)(?:^|\n).*Here'? (?:is|are) the extracted .*",  # match start of line or after newline
                    "",                                                    # remove it
                    collate_answer,
                    flags=re.MULTILINE  # allows ^ and $ to match line boundaries
                ).strip()
    
            # Remove any occurrence of "Here's the extracted requirement:" or variants anywhere
            collate_answer = collate_answer.replace("Here's the extracted requirement:", "")
            collate_answer = collate_answer.replace("Here is the extracted requirement:", "")
            collate_answer = collate_answer.replace("Here are the extracted requirement:", "")
            
            # Optional: also strip leftover spaces/newlines
            collate_answer = collate_answer.strip()

            match = re.search(r'Image:\s*(.*?\.png)', answer)
            if match:
                image_path = match.group(1)
                st.image(image_path, caption=f"------------")

        # --- generate tables ---
        st.session_state.c_df = collate_and_summarize.parse_requirements_to_df(collate_answer)

        # Now fill empty requirements from excerpt
        st.session_state.c_df['requirement'] = st.session_state.c_df.apply(
            lambda row: f"Excerpt from Source: {row['excerpt']}" 
                        if not row['requirement'] and row['excerpt'] 
                        else row['requirement'],
            axis=1
        )

        # sometimes LLM hallucinate, non deterministic, upper case just to make sure
        st.session_state.c_df["category"] = st.session_state.c_df["category"].str.upper()
        st.session_state.c_df["type"] = st.session_state.c_df["type"].str.upper()
        st.session_state.c_df["classification"] = st.session_state.c_df["classification"].str.upper()
        st.session_state.c_df["compliance"] = st.session_state.c_df["compliance"].str.upper()

        st.session_state.c_df_text = st.session_state.c_df[st.session_state.c_df["type"]=='TEXT']
        st.session_state.c_df_image = st.session_state.c_df[st.session_state.c_df["type"]=='IMAGE']
        
        st.session_state.summarize_c_df = collate_and_summarize.summarize_df(st.session_state.c_df)

# Display tables and download buttons outside the `if query:` block
# so buttons persist across reruns
if st.session_state.get("c_df_text") is not None:
    st.write("### Raw Requirements Table Text")
    st.dataframe(st.session_state.c_df_text)
    csv1_1 = to_csv_bytes(st.session_state.c_df_text)
    if st.download_button(
        label="Download Table 1_1",
        data=csv1_1,
        file_name="table1_1.csv",
        mime="text/csv",
        key="download_table1_1"
    ):
        st.session_state.download1_1_clicked = True
    if st.session_state.download1_1_clicked:
        st.success("Table 1_1 ready to download")

if st.session_state.get("c_df_text") is not None:
    st.write("### Raw Requirements Table Image")
    st.dataframe(st.session_state.c_df_image)
    csv1_2 = to_csv_bytes(st.session_state.c_df_image)
    if st.download_button(
        label="Download Table 1_2",
        data=csv1_2,
        file_name="table1_2.csv",
        mime="text/csv",
        key="download_table1_2"
    ):
        st.session_state.download1_2_clicked = True
    if st.session_state.download1_2_clicked:
        st.success("Table 1_2 ready to download")

if st.session_state.get("summarize_c_df") is not None:
    st.write("### Summarized Requirements Table")
    st.dataframe(st.session_state.summarize_c_df)
    csv2 = to_csv_bytes(st.session_state.summarize_c_df)
    if st.download_button(
        label="Download Table 2",
        data=csv2,
        file_name="table2.csv",
        mime="text/csv",
        key="download_table2"
    ):
        st.session_state.download2_clicked = True
    if st.session_state.download2_clicked:
        st.success("Table 2 ready to download")


if st.session_state.get("dframefinal") is not None:
    st.write("### Bill of Quantities Table")
    st.dataframe(st.session_state.dframefinal)
    csv2 = to_csv_bytes(st.session_state.dframefinal)
    if st.download_button(
        label="Download Table 3",
        data=csv2,
        file_name="table3.csv",
        mime="text/csv",
        key="download_table3"
    ):
        st.session_state.download3_clicked = True
    if st.session_state.download3_clicked:
        st.success("Table 3 ready to download")