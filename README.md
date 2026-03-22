### text_and_image_preprocessing.py

It automates a step-by-step preprocessing pipeline for text and images ingestion into a vector database with failure control.

- Creates a pipeline runner called text_and_image_preprocessing.py
- Defines a list of scripts to execute:
  - text_ingest.py
  - image_read.py
  - image_ingest.py
- Uses `subprocess.run()` to execute each script
- Runs scripts sequentially (one after another)
- Waits for each script to finish before starting the next
- Prints which script is currently running
- Checks execution status using returncode
- If a script fails `(returncode != 0):`
- Prints an error message
- Stops the pipeline immediately

### text_ingest.py

- Configuration & Setup
  - Loads configuration from `config.json`
    - `embedding_path` → word embeddings file
    - `docs_path` → folder containing PDFs
    - `vector_pickle_all` → saved vector database file
  - Imports required libraries
    - LangChain
    - NumPy
    - regex
    - Custom modules: `vector_db_lib`, `llm_tools`, `collate_and_summarize`, `table_process`
- Embedding Index Creation
  - Reads embedding file line by line
  - Creates a dictionary: `word → vector (numpy array)`
  - Used later for vectorization of text chunks
- PDF Loading
  - Recursively scans `docs_path` for all `.pdf` files
  - Uses `PyPDFLoader` to load pages
  - For each page:
    - Adds metadata:
      - `file` → filename
      - `path` → full path
  - Stores all pages into `docs` list
- Text Chunking
  - Uses `RecursiveCharacterTextSplitter`
    - `chunk_size = 1000`
    - `chunk_overlap = 200`
  - Splits documents into smaller chunks for easier processing
- Reformat to Records
  - Converts chunks into structured dictionary format:
    - `text → content`
    - `file → filename`
    - `page → page number`
    - `type → "text"`
  - Stores all records in `records` list
- Load Existing Vector DB
  - Loads existing vector database from pickle file
  - Prevents recomputation of already processed chunks
- Duplicate Prevention
  - Creates a set of unique keys: `(file, page, text, type)`
  - Ensures no duplicate entries are added
- Embedding Dimension
  - Determines vector length (`len_vec`) from embedding index
- Vectorization & Storage
  - Calls `add_to_vector_db()` to:
    - Convert text chunks into vectors
    - Add only new (non-duplicate) entries
  - Uses embedding index and metadata fields (`file`, `page`, `text`, `image`)
- Save Updated Database
  - Saves the updated vector database back to the pickle file
