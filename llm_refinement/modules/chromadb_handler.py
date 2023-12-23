import chromadb
from chromadb.config import Settings


class ChromaDBHandler:
    def __init__(self, persist_dir, collection_name):
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self.client = self.init_client()
        self.collection = self.load_collection()

    def init_client(self):
        client = chromadb.Client(Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory=self.persist_dir
        ))
        return client

    def load_collection(self):
        """
        Load the collection of clinical trials from the ChromaDB database.

        Returns:
            - chromadb.Collection: The ChromaDB collection
        """
        collection = self.client.get_collection(self.collection_name)
        return collection
