# ----------------------------------------------------------
# to infer description from image that is extracted from pdf
# main "llama3.2-vision" and fallback "qwen2.5vl:7b"
# output is image_analysis.json
# shazri 2026
# ----------------------------------------------------------

import fitz
import os
import base64
import json
import requests
from PIL import Image

import json

with open("config.json", "r") as f:
    config = json.load(f)

embedding_path = config["embedding_path"]

print("Embeddings:", embedding_path)

import os
import base64
import json
import fitz  # PyMuPDF
from PIL import Image
import requests

DOCS_FOLDER = config["docs_path"]
IMAGE_DIR = config["images_rep"]
image_analysis_json = config["image_analysis"]

os.makedirs(IMAGE_DIR, exist_ok=True)
results = []

# -----------------------------
# Resize image (speed boost)
# -----------------------------
def resize_image(path, max_size=512):
    try:
        img = Image.open(path)
        img.thumbnail((max_size, max_size))
        img.save(path)
    except Exception as e:
        print("Resize failed:", e)

# -----------------------------
# Clean repetition
# -----------------------------
def clean_repetition(text):
    lines = text.split(". ")
    seen = set()
    cleaned = []

    for line in lines:
        line = line.strip()
        if line and line not in seen:
            cleaned.append(line)
            seen.add(line)

    return ". ".join(cleaned)

# -----------------------------
# Analyze image with fallback
# -----------------------------
def analyze_image(image_path):
    print(f"\n Analyzing: {image_path}")

    with open(image_path, "rb") as f:
        img_base64 = base64.b64encode(f.read()).decode()

    prompt = """
Describe this diagram from a technical or tender document.

Focus on:
- requirement
- prerequisite
- equipment
- process flow
- engineering elements
- safety elements

Return a short description.
"""

    models = [
        "llama3.2-vision",
        "qwen2.5vl:7b"
    ]

    for model in models:
        try:
            print(f"\n Trying model: {model}")
            print(" Sending request...")

            r = requests.post(
                "http://localhost:11434/api/chat",
                json={
                    "model": model,
                    "messages": [
                        {
                            "role": "user",
                            "content": prompt,
                            "images": [img_base64]
                        }
                    ],
                    "stream": True,
                    "options": {   #  CONTROL GENERATION
                        "num_predict": 300,
                        "temperature": 0.3,
                        "repeat_penalty": 1.2
                    }
                },
                stream=True,
                timeout=300
            )

            full_response = ""
            print("📡 Streaming response:")

            for line in r.iter_lines():
                if line:
                    chunk = json.loads(line.decode("utf-8"))

                    if "message" in chunk and "content" in chunk["message"]:
                        token = chunk["message"]["content"]
                        print(token, end="", flush=True)
                        full_response += token

                        #  stop early if looping detected
                        if full_response.count("the diagram") > 5:
                            print("\n Repetition detected, stopping early")
                            break

            print("\n Done with", model)

            return clean_repetition(full_response.strip())

        except Exception as e:
            print(f"\n {model} failed:", e)

    return "ERROR"

# -----------------------------
# Process PDF
# -----------------------------
def process_pdf(pdf_path):
    print(f"\n Processing PDF: {pdf_path}")

    doc = fitz.open(pdf_path)

    for page_index, page in enumerate(doc):
        print(f"\nPage {page_index}")

        images = page.get_images(full=True)

        for img_index, img in enumerate(images):
            print(f"   Image {img_index}")

            xref = img[0]
            pix = fitz.Pixmap(doc, xref)

            if pix.n > 4:
                pix = fitz.Pixmap(fitz.csRGB, pix)

            filename = f"{IMAGE_DIR}/{os.path.basename(pdf_path)}_p{page_index}_img{img_index}.png"
            pix.save(filename)
            pix = None

            # Resize for speed
            resize_image(filename)

            # Check size
            with Image.open(filename) as img_obj:
                w, h = img_obj.size

            if w < 300 or h < 300:
                print(f"   Skipped small image ({w}x{h})")
                os.remove(filename)
                continue

            print(f"  Valid image ({w}x{h})")

            description = analyze_image(filename)

            results.append({
                "pdf": pdf_path,
                "page": page_index,
                "image": filename,
                "description": description
            })

# -----------------------------
# Main loop
# -----------------------------
for root, dirs, files in os.walk(DOCS_FOLDER):
    for file in files:
        if file.lower().endswith(".pdf"):
            pdf_path = os.path.join(root, file)
            process_pdf(pdf_path)


for item in results:
    if "description" in item and isinstance(item["description"], str):
        item["description"] = item["description"].replace("\xad", " ").replace("\xa0", " ")


# -----------------------------
# Save results
# -----------------------------
with open(image_analysis_json, "w") as f:
    json.dump(results, f, indent=2)

print("\n Done. Results saved to:", image_analysis_json)

print(results)