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