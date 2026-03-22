# ---------------------------------------------------------------------
# To extract, chunk, attribute (including image to description), call vectorization and store  
# This makes it easier as it runs as a simple workflow
# shazri 2026
# ---------------------------------------------------------------------

import subprocess

scripts = [
    "text_ingest.py",
    "image_read.py",
    "image_ingest.py"
]

for script in scripts:
    print(f"Running {script}...")
    
    result = subprocess.run(["python", script])
    
    if result.returncode != 0:
        print(f"Error in {script}, stopping pipeline.")
        break