### text_and_image_preprocessing.py

It automates a step-by-step preprocessing pipeline for text and images ingestion into a vector database with failure control.

- Creates a pipeline runner called text_and_image_preprocessing.py
- Defines a list of scripts to execute:
  - text_ingest.py
  - image_read.py
  - image_ingest.py
- Uses subprocess.run() to execute each script
- Runs scripts sequentially (one after another)
- Waits for each script to finish before starting the next
- Prints which script is currently running
- Checks execution status using returncode
- If a script fails (returncode != 0):
- Prints an error message
- Stops the pipeline immediately
