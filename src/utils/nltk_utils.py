import nltk

from src.logger.custom_logger import logger


REQUIRED_RESOURCES = {
    "tokenizers/punkt": "punkt",
    "tokenizers/punkt_tab": "punkt_tab",
    "corpora/stopwords": "stopwords",
    "corpora/wordnet": "wordnet",
    "corpora/omw-1.4": "omw-1.4",
}


def download_nltk_resources() -> None:
    for resource_path, resource_name in REQUIRED_RESOURCES.items():
        try:
            nltk.data.find(resource_path)
            logger.info(f"NLTK resource already available: {resource_name}")

        except LookupError:
            logger.info(f"Downloading NLTK resource: {resource_name}")
            nltk.download(resource_name, quiet=True)

    logger.info("All required NLTK resources are available.")