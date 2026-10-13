import sqlite3
import os

DB_PATH = 'tickets.db'

def get_db_connection():
    """Establishes and returns a connection to the SQLite database."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Initializes the database and creates the tickets table if it does not exist."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS tickets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                complaint_text TEXT NOT NULL,
                category TEXT NOT NULL,
                priority TEXT NOT NULL,
                status TEXT DEFAULT 'Open',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        conn.commit()
    print("Database initialized successfully.")

def create_ticket(complaint_text, category, priority):
    """Inserts a new ticket into the database."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO tickets (complaint_text, category, priority, status)
            VALUES (?, ?, ?, 'Open')
        ''', (complaint_text, category, priority))
        conn.commit()
        return cursor.lastrowid

def get_all_tickets():
    """Retrieves all tickets from the database."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM tickets ORDER BY created_at DESC')
        return [dict(row) for row in cursor.fetchall()]

def update_ticket_status(ticket_id, new_status):
    """Updates the status of an existing ticket."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            UPDATE tickets
            SET status = ?
            WHERE id = ?
        ''', (new_status, ticket_id))
        conn.commit()
        return cursor.rowcount > 0

def delete_ticket(ticket_id):
    """Deletes a ticket from the database by ID."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('DELETE FROM tickets WHERE id = ?', (ticket_id,))
        conn.commit()
        return cursor.rowcount > 0

if __name__ == '__main__':
    init_db()
    # Test database creation
    test_id = create_ticket("Test complaint text", "Billing", "Low")
    print(f"Inserted test ticket ID: {test_id}")
    tickets = get_all_tickets()
    print(f"Total tickets: {len(tickets)}")
    update_ticket_status(test_id, "In Progress")
    print("Updated ticket status to 'In Progress'.")
    delete_ticket(test_id)
    print("Deleted test ticket.")
