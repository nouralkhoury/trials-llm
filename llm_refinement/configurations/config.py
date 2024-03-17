from starlette.config import Config

try:
    config = Config(".env")
except FileNotFoundError:
    config = Config()


# OpenAI API key
OPENAI_API_KEY = config("OPENAI_API_KEY", default="")

# PromptLayer API Key
PROMPTLAYER_API_KEY = config("PROMPTLAYER_API_KEY", default="")


# Data Directories
PROCESSED_DATA = config("PROCESSED_DATA", default="data/processed")
RESULTS_DATA = config("RESULTS_DATA", default="data/results")


# ChromaDB
CTRIALS_COLLECTION = config("CTRIALS_COLLECTION", default="ctrials")
CTRIALS_COLLECTION_TRAIN = config("CTRIALS_COLLECTION_TRAIN", default="train-ctrials")
PERSIST_DIRECTORY = config("PERSIST_DIRECTORY", default="data/collections")
PERSIST_DIRECTORY_TRAIN = config("PERSIST_DIRECTORY", default="data/collection_train")
