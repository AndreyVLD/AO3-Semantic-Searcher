# AO3 Semantic Search

This project provides a semantic search engine for fanfiction
from [Archive of Our Own (AO3)](https://archiveofourown.org/). It uses a
retrieve-and-rerank pipeline with sentence-transformer models to find works based on the semantic meaning of a user's
query, rather than just keyword matching. The user interface is built with Streamlit.

The fanfiction data used to build the ao3_current.sqlite3 database can be found in dataset provided by the [AO3 final
location data dump](https://archive.org/details/AO3_final_location) provided by Internet Archive. This database contains
metadata for 15952441 entries from which 10281597 are unique works as of April 2023.

## Project Structure

```
.
├── data/
│   └── ao3_current.sqlite3     # SQLite database with works and embeddings
├── scripts/
│   └── create_embeddings.py     # Script to generate embeddings for works
├── src/
│   ├── config.py               # Configuration for models, paths, and constants
│   ├── db.py                   # Database interaction logic
│   └── embedding.py            # Handles embedding and cross-encoding models
└── app.py                      # The main Streamlit application
```

## How to Run

### Prerequisites

- Python 3.13 or higher
- A virtual environment (recommended)
- Preferably a GPU for faster inference (optional, but recommended for faster embeddings generation and reranking)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/AndreyVLD/AO3-Semantic-Searcher
   cd AO3-Semantic-Searcher
    ```
2. Create and activate a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```
3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```
4. Download the AO3 database file `ao3_current.sqlite3` and place it in the `data/` directory.
   You can find the database in the [AO3 final location data dump](https://archive.org/details/AO3_final_location).

### Step 1: Generate Embeddings

Before you can search, you need to populate the database with vector embeddings for the fanfiction works. The script is
located in `scripts/create_embeddings.py`.

Run the script to generate embeddings for all works in the database:

```bash
python -m scripts.create_embeddings
```

### Step 2: Run the Search App

Once the embeddings are generated, you can launch the search application.

1. Run the Streamlit app from your terminal:
    ``` bash
    streamlit run app.py
    ```
2. Open your web browser to the local URL provided by Streamlit.
3. Type your search query into the text box and click "Submit" to find relevant fanfiction. Works best with natural
   language queries.