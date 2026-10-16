import sqlite3

DB_NAME = "mood_tracker.db"


def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXIST mood_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            emotion TEXT NOT NULL,
            confidence REAL NOT NULL
        )
    """)
    conn.commit()
    conn.close()


def log_mood(emotion: str, confidence: float):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO mood_logs (emotion, confidence)
        VALUES (?, ?)
    """,
        (emotion, confidence),
    )
    conn.commit()
    conn.close()


if __name__ == "__main__":
    init_db()
    print("SQLite database successfully initialized.")
