import os
import sys
import logging

LOG_FORMAT = (
    "[%(asctime)s] | %(levelname)-8s | %(name)s | "
    "%(module)s:%(lineno)d | %(message)s"
)

LOG_DIR = "logs"
LOG_FILE = "application.log"

os.makedirs(LOG_DIR, exist_ok=True)

LOG_FILE_PATH = os.path.join(LOG_DIR, LOG_FILE)

logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(LOG_FILE_PATH, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)

logger = logging.getLogger("NNClogger")