import sys
import yaml
import pandas as pd

from src.logger.custom_logger import logger
from src.exception.custom_exception import CustomException

df = pd.read_csv("data/raw/train.csv")

schema = {
    "COLUMNS":{
        col: str(dtype)
        for col, dtype in df.dtypes.items()
    },
    
    "TARGET_COLUMN":{
        "target":"label"
    }
}


try: 
    with open ("config/schema.yaml", "w") as file:
        yaml.dump(schema, file)

    logger.info("schema.yaml generated successfully")
except Exception as e:
    raise CustomException(e, sys)