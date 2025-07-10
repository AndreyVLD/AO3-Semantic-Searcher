import sqlite3
import sqlite_vec

from dataclasses import dataclass
from typing import Optional, Iterator
from numpy import ndarray


@dataclass()
class Work:
    path: str
    title: str
    author: str
    category: str
    genre: str
    rating: str
    warnings: str
    summary: Optional[str] = None
    storyURL: Optional[str] = None
    relationships: Optional[str] = None
    series: Optional[str] = None
    collections: Optional[str] = None

    def __repr__(self) -> str:
        return self.get_embedding_text()

    def get_embedding_text(self) -> str:
        """
        Construct a labeled, separator-delimited text blob for embedding.
        """

        # Map field names to their labels and values
        field_map = {
            "TITLE": self.title,
            "AUTHOR": self.author,
            "CATEGORY": self.category,
            "GENRE": self.genre,
            "RATING": self.rating,
            "WARNINGS": self.warnings,
            # Optional fields
            "RELATIONSHIPS": self.relationships,
            "SUMMARY": self.summary,
            "SERIES": self.series,
            "COLLECTIONS": self.collections,
        }

        # Build lines only for non-empty values
        lines = []
        for label, value in field_map.items():
            if value:
                lines.append(f"{label}: {value.strip()}")

        return "\n\n".join(lines)


@dataclass
class RetrievedWork(Work):
    score: float = 0.0  # Default score will be updated after Re-Ranking


class WorkRepository:
    def __init__(self, db_path: str) -> None:
        self.connection = sqlite3.connect(db_path)
        self.connection.enable_load_extension(True)
        sqlite_vec.load(self.connection)
        self._create_embeddings_table()
        self._create_index()

    def _create_embeddings_table(self) -> None:
        cursor = self.connection.cursor()
        query = """
        CREATE VIRTUAL TABLE IF NOT EXISTS embeddings_bi_encoder 
        USING vec0(
            path TEXT PRIMARY KEY,
            embedding float[384],
        )
        """
        cursor.execute(query)
        self.connection.commit()

    def _create_index(self) -> None:
        cursor = self.connection.cursor()

        query = """
                CREATE INDEX IF NOT EXISTS idx_metadata_language
                    ON metadata (language);
                """
        cursor.execute(query)
        query = """
                CREATE INDEX IF NOT EXISTS idx_metadata_path
                    ON metadata (path);
                """
        cursor.execute(query)

        query = """
                CREATE INDEX IF NOT EXISTS idx_metadata_storyURL
                    ON metadata (storyURL);
                """
        cursor.execute(query)

        self.connection.commit()

    def get_count(self) -> int:
        cursor = self.connection.cursor()
        query = "SELECT COUNT(*) FROM metadata WHERE language = 'English'"
        cursor.execute(query)
        count, = cursor.fetchone()
        return count

    def get_works(self, batch_size: int = 1000) -> Iterator[list[Work]]:
        cursor = self.connection.cursor()

        query = """
                SELECT path,
                       title,
                       author,
                       category,
                       genre,
                       rating,
                       warnings,
                       summary,
                       storyURL,
                       relationships,
                       series,
                       collections
                FROM metadata
                WHERE language = 'English' \
                """
        cursor.execute(query)

        while True:
            rows = cursor.fetchmany(batch_size)

            if not rows:
                break

            works = [Work(*row) for row in rows]
            yield works

    def retrieve_top_k_works(self, embedding: ndarray, top_k: int) -> list[RetrievedWork]:
        cursor = self.connection.cursor()
        query = """
                WITH top_k AS (SELECT path, vec_distance_cosine(embedding, ?) AS score
                               FROM embeddings_bi_encoder
                               ORDER BY score
                               LIMIT ?)
                SELECT t.path,
                       m.title,
                       m.author,
                       m.category,
                       m.genre,
                       m.rating,
                       m.warnings,
                       m.summary,
                       m.storyURL,
                       m.relationships,
                       m.series,
                       m.collections,
                       t.score
                FROM top_k AS t
                         JOIN metadata AS m
                              ON t.path = m.path
                ORDER BY t.score;
                """
        cursor.execute(query, (embedding, top_k))
        rows = cursor.fetchall()
        works = [RetrievedWork(*row) for row in rows]

        return works

    def insert_embeddings(self, embeddings_batch: list[tuple[str, ndarray]]) -> None:
        cursor = self.connection.cursor()
        query = "REPLACE INTO embeddings_bi_encoder (path, embedding) VALUES (?, ?)"
        cursor.executemany(query, embeddings_batch)
        self.connection.commit()

    def drop_table(self, table_name: str) -> None:
        cursor = self.connection.cursor()
        query = "DROP TABLE IF EXISTS {}".format(table_name)
        cursor.execute(query)
        self.connection.commit()

    def remove_duplicate_works(self) -> None:
        cursor = self.connection.cursor()
        cursor.execute("""
                       SELECT COUNT(DISTINCT m.storyURL) AS expected_count
                       FROM embeddings_bi_encoder AS e
                                JOIN metadata AS m
                                     ON e.path = m.path
                       WHERE m.storyURL IS NOT NULL;
                       """)
        expected_count, = cursor.fetchone()

        try:
            cursor.execute("BEGIN TRANSACTION;")
            cursor.execute("""
                           -- 1) Build and rank candidates by storyURL & Packaged date
                           WITH ranked AS (SELECT e.path,
                                                  m.storyURL,
                                                  m.Packaged,
                                                  ROW_NUMBER() OVER (
                                                      PARTITION BY m.storyURL
                                                      ORDER BY m.Packaged DESC
                                                      ) AS rn
                                           FROM embeddings_bi_encoder AS e
                                                    JOIN metadata AS m
                                                         ON e.path = m.path
                                           WHERE m.storyURL IS NOT NULL),

                                duplicates AS (SELECT path
                                               FROM ranked
                                               WHERE rn > 1)

                           -- 2) Delete all but the newest (rn > 1) per storyURL
                           DELETE
                           FROM embeddings_bi_encoder
                           WHERE path IN duplicates
                           """)
            cursor.execute("SELECT COUNT(*) FROM embeddings_bi_encoder;")
            count, = cursor.fetchone()

            if expected_count - 10 <= count <= expected_count + 10:  # Allow some leeway for NULL storyURLs
                self.connection.commit()
                print(f"Removed duplicates successfully. Expected works: {expected_count}. Remaining works: {count}.")
            else:
                self.connection.rollback()
                print(f"Expected {expected_count} works, but found {count}. Rolling back changes.")

        except Exception as e:
            self.connection.rollback()
            print(f"Error removing duplicate works:\n{e}")

    def close(self) -> None:
        self.connection.close()
