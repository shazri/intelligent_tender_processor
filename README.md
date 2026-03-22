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
    - `embedding_path` , word embeddings file
    - `docs_path` , folder containing PDFs
    - `vector_pickle_all` , saved vector database file
  - Imports required libraries
    - LangChain
    - NumPy
    - regex
    - Custom modules: `vector_db_lib`, `llm_tools`, `collate_and_summarize`, `table_process`
- Embedding Index Creation
  - embedding token by token, and averaging representation
  - Creates a dictionary: `word to vector (numpy array)`
  - Used later for vectorization of text chunks
- PDF Loading
  - Recursively scans `docs_path` for all `.pdf` files
  - Uses `PyPDFLoader` to load pages
  - For each page:
    - Adds metadata:
      - `file` , filename
      - `path` , full path
  - Stores all pages into `docs` list
- Text Chunking
  - Uses `RecursiveCharacterTextSplitter`
    - `chunk_size = 1000`
    - `chunk_overlap = 200`
  - Splits documents into smaller chunks for easier processing
- Reformat to Records
  - Converts chunks into structured dictionary format:
    - `text , content`
    - `file , filename`
    - `page , page number`
    - `type , "text"`
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

### image_read.py

- Code Functionalities
- Reads config paths
  - PDF folder
  - Image output folder
  - JSON output file
- Walks through all folders and PDFs
  - Uses `os.walk`
  - Filters `.pdf` files
- Extracts images from PDFs
  - Uses `fitz` (PyMuPDF)
  - Loops page by page
  - Extracts all images
- Converts and saves images
  - Ensures RGB format
  - Saves as `.png`
  - Names include file, page, image index
- Filters images
  - Removes small images (<300px)
  - Avoids noise like icons/logos
- Resizes images
  - Max 512px
  - Improves speed and reduces load
- Sends image to Ollama
  - Base64 encoding
  - Uses `/api/chat`
  - Prompt focuses on:
    - Requirement
    - Equipment
    - Process
    - Safety
- Uses model fallback
  - First: `llama3.2-vision`
  - Fallback: `qwen2.5vl:7b`
- Controls model output
  - `num_predict = 300`
  - `temperature = 0.3`
  - `repeat_penalty = 1.2`
- Streams response
  - Prints tokens live
  - No waiting blindly
- Handles repetition
  - Detects looping text
  - Stops early if repeating
  - Cleans duplicate sentences
- Error handling
  - Catches request failures
  - Tries next model
  - Returns `"ERROR"` if all fail
- Stores results
  - List of dictionaries
  - Fields:
    - `pdf`
    - `page`
    - `image`
    - `description`
- Saves output
  - Writes to JSON file


### image_ingest.py

- Load Configuration
  - Reads paths from `config.json`:
    - `embedding_path` → word embeddings file
    - `docs_path` → folder containing PDFs
    - `vector_pickle_all` → saved vector database
    - `image_analysis` → JSON file with image analysis results
  - Prints configuration paths

- Load Image Records
  - Reads `image_analysis` JSON
  - Prints total number of records and sample record

- Create Embedding Index
  - Reads embeddings file line by line
  - Creates dictionary:
    - `word` → vector (`numpy` array)
  - Used for vectorization of text/image descriptions

- Load Existing Vector Database
  - Loads vector DB from pickle file
  - Creates a set of unique keys:
    - `(file, page, text, type)`
  - Prevents adding duplicates

- Determine Embedding Dimension
  - Gets vector length from embedding index

- Add Image Records to Vector DB
  - Calls `add_to_vector_db()` with:
    - Records: `image_records`
    - Fields: `description`, `pdf`, `page`, `image`, type=`image`
    - Existing keys to avoid duplicates
    - Embedding index and vector length

- Save Updated Vector Database
  - Saves updated DB back to pickle file
  - Prints total number of vectors
 

### llm_tools.py

- LLM Tools Library
  - Provides reusable functions for embedding, vector search, and querying LLMs (Ollama)
  - Supports requirement extraction from tender documents

- Configuration
  - Loads paths from `config.json`:
    - `embedding_path` → word embeddings file
    - `docs_path` → folder containing PDFs
    - `vector_pickle_all` → saved vector database file

- LLM Interaction
  - `ask_ollama(prompt)`
    - Sends a POST request to Ollama API (`llama3`) at `localhost:11434`
    - Options:
      - `temperature`: 0
      - `top_p`: 1
      - `top_k`: 1
    - Returns LLM response as JSON

  - `ask_requirements(context)`
    - Wraps context into a strict prompt for requirement extraction
    - Rules enforced:
      - No summarization
      - Copy file and page metadata exactly
      - Only allowed categories, classification, compliance
      - Returns formatted requirement output

- Embedding Utilities
  - Loads embeddings from `embedding_path` into a dictionary:
    - `word` to vector (`numpy` array)
  - Determines embedding dimension (`len_vec`) automatically

- Vector Database Interaction
  - Loads vector database from pickle file (`vector_db_lib.load_vector_db`)

- Vector Search
  - `vectorize(text)`
    - Converts input text into vector representation
    - Uses preloaded embedding index
    - Averages word vectors for multi-token text

  - `cosine(a, b)`
    - Computes cosine similarity between two vectors

  - `search(query_vector, k=5)`
    - Returns top `k` most similar items from the vector database
    - Sorts by cosine similarity descending
   
  ### collate_and_summarize.py

  - Collate and Summarize Library
  - Provides functions to parse and summarize LLM-generated requirement text
  - Converts raw requirement strings into structured Pandas DataFrames

- Parser: `parse_requirements_to_df(text)`
  - Cleans LLM-generated fluff such as:
    - "Here is the extracted …"
    - "Here's the extracted …"
    - "Here are the extracted …"
  - Extracts structured fields for each requirement:
    - requirement
    - excerpt
    - category
    - file
    - page
    - type
    - image
    - classification
    - compliance
  - Combines multiple excerpts into one
  - Fills empty `requirement` fields from `excerpt` with tracking

- Summarizer: `summarize_df(c_df)`
  - Aggregates the DataFrame by:
    - category
    - classification
    - compliance
  - Produces a side-by-side summary per file
  - Cleans up duplicated file names
  - Replaces missing values with empty strings
  - Returns a final, clean summary table
