# 🧬 [QUANTUM_EVOLUTION]: Gen_9 Linked
import telemetry_bridge
import os
import sqlite3


class DatabaseRecovery:

    def __init__(self, db_name="agi_system.db"):
        """
        Initialize the DatabaseRecovery class.

        :param db_name: The name of the database file.
        """
        self.db_name = db_name
        self.journal_file = f"{db_name}-journal"
        self.wal_file = f"{db_name}-wal"
        self.shm_file = f"{db_name}-shm"

    def is_journal_file_present(self):
        """
        Check if the database journal file exists.

        :return: True if the journal file exists, False otherwise.
        """
        return os.path.exists(self.journal_file)

    def is_wal_file_present(self):
        """
        Check if the database WAL file exists.

        :return: True if the WAL file exists, False otherwise.
        """
        return os.path.exists(self.wal_file)

    def is_shm_file_present(self):
        """
        Check if the database SHM file exists.

        :return: True if the SHM file exists, False otherwise.
        """
        return os.path.exists(self.shm_file)

    def remove_journal_file(self):
        """
        Remove the database journal file to recover the database.
        """
        if self.is_journal_file_present():
            os.remove(self.journal_file)

    def remove_wal_file(self):
        """
        Remove the database WAL file to recover the database.
        """
        if self.is_wal_file_present():
            os.remove(self.wal_file)

    def remove_shm_file(self):
        """
        Remove the database SHM file to recover the database.
        """
        if self.is_shm_file_present():
            os.remove(self.shm_file)

    def connect_to_database(self):
        """
        Attempt to connect to the recovered database.

        :return: True if the connection is successful, False otherwise.
        """
        try:
            conn = sqlite3.connect(self.db_name)
            conn.close()
            return True
        except sqlite3.Error as e:
            print(f"Error recovering database: {e}")
            return False

    def recover(self):
        """
        Recover the database by removing the journal, WAL, and SHM files and checking the database connection.
        """
        if self.is_journal_file_present():
            self.remove_journal_file()
        if self.is_wal_file_present():
            self.remove_wal_file()
        if self.is_shm_file_present():
            self.remove_shm_file()
        if self.connect_to_database():
            print("Database recovery successful")
        else:
            print("Database recovery failed")


if __name__ == "__main__":
    db_recovery = DatabaseRecovery()
    db_recovery.recover()
