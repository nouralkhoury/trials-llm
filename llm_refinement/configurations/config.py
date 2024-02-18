from starlette.config import Config
from starlette.datastructures import Secret


try:
    config = Config(".env")
except FileNotFoundError:
    config = Config()


# OpenAI API key
OPENAI_API_KEY = config("OPENAI_API_KEY", default="", cast=Secret)

# PromptLayer API Key
PROMPTLAYER_API_KEY = config("PROMPTLAYER_API_KEY", default="", cast=Secret)


# Data Directories
PROCESSED_DATA = config("PROCESSED_DATA", default="data/processed")
RESULTS_DATA = config("RESULTS_DATA", default="data/results")


# ChromaDB
CTRIALS_COLLECTION = config("CTRIALS_COLLECTION", default="ctrials")
PERSIST_DIRECTORY = config("PERSIST_DIRECTORY", default="data/collections")
