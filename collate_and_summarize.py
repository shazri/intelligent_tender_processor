# --------------------------------------------------------------
# from output of LLM and auxillary system, generate dataframe 
# generate summarization
# shazri 2026
# --------------------------------------------------------------

import pandas as pd
import re

# -------------------------
# parser unto strings gemerated by 
# LLM to generate dataframe
# -------------------------

def parse_requirements_to_df(text):
    """
    Parse LLM-generated requirements text into a clean DataFrame.
    Removes any 'Here is / Here's / Here are the extracted ...' fluff.
    """

    # ----------------------------
    # 1. Remove global LLM fluff at the very start of text
    # ----------------------------
    text = re.sub(
        r"(?i)\s*Here'? (is|are) the extracted [^\n]*",
        "",
        text,
        flags=re.MULTILINE
    )

    # ----------------------------
    # 2. Regex pattern for each requirement block
    # ----------------------------
    pattern = re.findall(
        r"Requirement:\s*(.*?)\n"                     # requirement
        r"(?:\s*Excerpt from Source:\s*(.*?)\n)*"    # optional excerpt 1
        r"Category of Requirement:\s*(.*?)\s*\n"
        r"(?:\s*Excerpt from Source:\s*(.*?)\n)*"    # optional excerpt 2
        r"File:\s*(.*?)\s*\n"
        r"Page:\s*(.*?)\s*\n"
        r"Type:\s*(.*?)\s*\n"
        r"Image:\s*(.*?)\s*\n"
        r"Classification:\s*(.*?)\s*\n"
        r"Compliance:\s*(.*?)(?=\nRequirement:|\Z)", # stop at next Requirement or end
        text,
        flags=re.DOTALL | re.IGNORECASE
    )

    rows = []

    # ----------------------------
    # 3. Function to remove fluff anywhere in a string
    # ----------------------------
    def remove_fluff(field):
        if not field:
            return ""
        # remove any "Here is / Here's / Here are the extracted ..." anywhere
        return re.sub(r"(?i)\s*Here'? (is|are) the extracted .*", "", field).strip()

    for match in pattern:
        requirement, excerpt1, category, excerpt2, file, page, typ, image, classification, compliance = match

        # Combine excerpts
        excerpt = "\n".join(filter(None, [excerpt1, excerpt2]))

        # Clean fluff in every relevant field
        requirement = remove_fluff(requirement)
        excerpt     = remove_fluff(excerpt)
        category    = remove_fluff(category)
        compliance  = remove_fluff(compliance)

        # If requirement is empty/None, fill from excerpt with tracking
        if not requirement and excerpt:
            requirement = f"Excerpt from Source: {excerpt}"

        rows.append({
            "requirement": requirement,
            "excerpt": excerpt,
            "category": category,
            "file": file.strip(),
            "page": page.strip(),
            "type": typ.strip(),
            "image": image.strip(),
            "classification": classification.strip(),
            "compliance": compliance
        })

    return pd.DataFrame(rows)
    
# -------------------------
# summarize the dataframe results
# -------------------------

def summarize_df(c_df):
    # --- category ---
    cat = (
        c_df.groupby(["file", "category"])
        .size()
        .reset_index(name="cat_count")
    )
    cat["row"] = cat.groupby("file").cumcount()
    
    # --- classification ---
    cls = (
        c_df.groupby(["file", "classification"])
        .size()
        .reset_index(name="class_count")
    )
    cls["row"] = cls.groupby("file").cumcount()
    
    # --- compliance ---
    comp = (
        c_df.groupby(["file", "compliance"])
        .size()
        .reset_index(name="comp_count")
    )
    comp["row"] = comp.groupby("file").cumcount()
    
    # --- merge side-by-side ---
    summary_c_df = cat.merge(cls, on=["file", "row"], how="outer") \
                      .merge(comp, on=["file", "row"], how="outer")
    
    # --- clean column names ---
    summary_c_df = summary_c_df.rename(columns={
        "category": "category",
        "classification": "classification",
        "compliance": "compliance"
    })
    
    # --- sort by file and row ---
    summary_c_df = summary_c_df.sort_values(["file", "row"]).drop(columns="row")
    
    # --- merged file look ---
    summary_c_df["file"] = summary_c_df["file"].mask(summary_c_df["file"].duplicated())
    
    # --- replace NaN with empty string ---
    summary_c_df = summary_c_df.fillna("")
    
    # --- final clean table ---
    return summary_c_df
