import pandas as pd
import numpy as np
import pdfplumber


def apply_left_fill_rule(df):
    """
    Rule:
    - For each column in df:
        - If column name is not empty/null
        - And all values are empty or null
        - Take content from the column immediately to the left
    """
    df = df.copy()

    for i, col in enumerate(df.columns):
        if i == 0:
            continue  # skip first column

        # Check column name is not empty/null
        if pd.notna(col) and str(col).strip() != "":
            # Check if all values are empty or null
            col_values = df[col].apply(lambda x: str(x).strip() if pd.notna(x) else "")
            if col_values.eq("").all():
                left_col = df.columns[i - 1]
                
                # Ensure left column is a Series, align length
                if isinstance(df[left_col], pd.DataFrame):
                    left_series = df[left_col].iloc[:,0]
                else:
                    left_series = df[left_col]

                # Assign safely
                df[col] = pd.Series(left_series.values, index=df.index)

    return df



def extract_boq(text):
    import requests
    import json
    import re

    prompt = f"""
You are an expert in extracting Bill of Quantities (BoQ) from engineering tender text.

Rules:
- Extract ONLY explicit items with quantities.
- Do NOT summarize.
- Do NOT include general sentences.
- Each row must have:
  Description | Unit | Quantity | Notes
- Description must match text exactly
- Unit must be normalized (Nos or No.)
- Quantity must be numeric only
- Notes can capture context (e.g., section i), ii), iii), AND inferred purpose be elaborative with short paragraph)
- Ignore headings like i), ii), iii) as separate rows, but use them in Notes

Return ONLY JSON array like:
[
  {{
    "Description": "...",
    "Unit": "...",
    "Quantity": "...",
    "Notes": "..."
  }}
]

Text:
{text}
"""

    r = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "llama3",
            "prompt": prompt,
            "stream": False,
            "temperature": 0
        }
    )

    response = r.json()["response"]

    # Extract JSON safely
    match = re.search(r"\[.*\]", response, re.DOTALL)
    if match:
        return json.loads(match.group())
    else:
        print("Failed to extract JSON")
        print(response)
        return []

def call_ollama(prompt):
    import requests

    r = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "llama3",
            "prompt": prompt,
            "stream": False,
            "temperature": 0
        }
    )

    return r.json()["response"]




def process_table():

    """
    ingest tables across pages
    """
    
    pdf_file = "docs//Volume-III//1. Bid Price Schedule.pdf"
    all_tables = []
    
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            table = page.extract_table()
            if table and len(table) > 2:
                # Skip first 2 rows
                df_page = pd.DataFrame(table[2:], columns=table[0])
                
                # Drop rows where 'Description/Scope of Work' is empty or None
                df_page = df_page[df_page['Description/Scope of Work'].notna() & 
                                  (df_page['Description/Scope of Work'] != '')]
                
                all_tables.append(df_page)
    
    # Combine all pages into one DataFrame
    if all_tables:
        df_full = pd.concat(all_tables, ignore_index=True)
    else:
        df_full = pd.DataFrame()
    

    """
    use rule that , when column is not empty and all the content is empty
    then, take all the one-left values to column 
    """
    
    df_full = apply_left_fill_rule(df_full)
    
    # Drop columns where column name is None or empty
    df_full = df_full.loc[:, [col for col in df_full.columns if col not in [None, '']]]
    
    
    # Drop columns where all values are NaN
    df_full = df_full.dropna(axis=1, how='all')

    """
    if S. , Unit and Quantity empty merge below content with above content
    """
    
    cleaned_rows = []
    prev_row = None
    
    for _, row in df_full.iterrows():
        # Check if it's a continuation row
        if (
            (pd.isna(row["S."]) or row["S."] == "") and
            (pd.isna(row["Unit"]) or row["Unit"] == "") and
            (pd.isna(row["Quantity"]) or row["Quantity"] == "")
        ):
            # Merge column by column into previous row
            if prev_row is not None:
                for col in df_full.columns:
                    if pd.notna(row[col]) and row[col] != "":
                        prev_row[col] = str(prev_row[col]) + " " + str(row[col])
        else:
            # New valid row
            if prev_row is not None:
                cleaned_rows.append(prev_row)
            prev_row = row.copy()
    
    # Append last row
    if prev_row is not None:
        cleaned_rows.append(prev_row)
    
    # Create new DataFrame
    df_clean = pd.DataFrame(cleaned_rows).reset_index(drop=True)

    """
    if the Unit and Quantity is empty , break-down description to 
    Bill of Quantities (BoQ)
    
    else, infer note from previous rows as context
    """
    
    final_rows = []
    context_memory = ""  # stores previous descriptions
    
    for _, row in df_clean.iterrows():
    
        desc = str(row["Description/Scope of Work"])
        unit = str(row["Unit"])
        qty  = str(row["Quantity"])
        itemno = str(row["S."])
    
        # Check if both Unit and Quantity are empty
        is_empty = (
            unit.strip() == "" or unit.lower() == "none"
        ) and (
            qty.strip() == "" or qty.lower() == "none"
        )
    
        # CASE 1: Extract BoQ
        if is_empty:
            boq_items = extract_boq(desc)
    
            for item in boq_items:
                final_rows.append({
                    "Item No." : itemno,
                    "Description": item.get("Description", ""),
                    "Unit": item.get("Unit", ""),
                    "Quantity": item.get("Quantity", ""),
                    "Notes": item.get("Notes", "")
                })
    
        # CASE 2: Generate Notes only
        else:
            note_prompt = f"""
    You are an engineering assistant.
    
    Generate a short note (max 8 words) for this item using previous context.
    
    Context:
    {context_memory}
    
    Current Description:
    {desc}
    
    Return ONLY the note.
    """
    
            note = call_ollama(note_prompt).strip()
    
            final_rows.append({
                "Item No." : itemno,
                "Description": desc,
                "Unit": unit,
                "Quantity": qty,
                "Notes": note
            })
    
            # update context AFTER using it
            context_memory += " " + desc
    
    
    # Final DataFrame
    df_final = pd.DataFrame(final_rows).reset_index(drop=True)
    
    return df_final
    

