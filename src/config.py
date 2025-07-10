from pathlib import Path

# Paths to existing and new DBs
WORKS_DB = Path(__file__).parents[1] / "data" / "ao3_current.sqlite3"

# Embedding models for the Retrieve & Re-Rank, `embeddings_bi_encoder` table
BI_ENCODER = "multi-qa-MiniLM-L6-cos-v1"
CROSS_ENCODER = "cross-encoder/ms-marco-MiniLM-L6-v2"

# Embedding model for the `embeddings` table
MODEL_NAME = "msmarco-MiniLM-L6-cos-v5"

# Batch sizes
DB_BATCH_SIZE = 10000
MODEL_BATCH_SIZE = 32

# Number of top works to retrieve
TOP_K = 32
