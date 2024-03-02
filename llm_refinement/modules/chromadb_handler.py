import chromadb
from chromadb.config import Settings
import logging


class ChromaDBHandler:
    def __init__(self, persist_dir, collection_name):
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self.client = self.init_client()
        self.collection = self.load_collection()

    def init_client(self):
        try:
            client = chromadb.Client(Settings(
                chroma_db_impl="duckdb+parquet",
                persist_directory=self.persist_dir
            ))
            return client
        except Exception as e:
            logging.error(f"Error initializing ChromaDB client: {e}")
            # Exit the program with a non-zero status code
            exit(1)

    def load_collection(self):
        try:
            """
            Load the collection of clinical trials from the ChromaDB database.

            Returns:
                - chromadb.Collection: The ChromaDB collection
            """
            collection = self.client.get_collection(self.collection_name)
            return collection
        except Exception as e:
            logging.error(f"Error loading ChromaDB collection: {e}")
            # Exit the program with a non-zero status code
            exit(1)

    def create_collection(self):
        try:
            collection = self.client.create_collection(name=self.load_collection)
            return collection
        except Exception as e:
            logging.error(f"Error creating ChromaDB collection: {e}")
            # Exit the program with a non-zero status code
            exit(1)
