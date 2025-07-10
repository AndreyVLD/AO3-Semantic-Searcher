import numpy as np

from tqdm import tqdm
from src.embedding import EmbeddingModel
from src.config import WORKS_DB, DB_BATCH_SIZE
from src.db import WorkRepository


def main() -> None:
    db = WorkRepository(WORKS_DB)

    model = EmbeddingModel()

    bar = tqdm(total=db.get_count(), desc="Creating embeddings for works", unit="work")

    # Iterate over works in the repository
    for batch in db.get_works(batch_size=DB_BATCH_SIZE):

        # Embed the text of each work in the batch
        texts = [str(work) for work in batch]
        embeddings = model.embed_chunks(texts)

        # Convert embeddings to a list of tuples (path, embedding)
        records = []
        for work, embedding in zip(batch, embeddings):
            record = (work.path, embedding.astype(np.float32))
            records.append(record)

        # Insert the embeddings into the database
        db.insert_embeddings(records)

        bar.update(len(batch))

    bar.close()

    # Removing duplicate embeddings
    db.remove_duplicate_works()

    db.close()


if __name__ == '__main__':
    main()
