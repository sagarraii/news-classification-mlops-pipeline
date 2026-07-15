import sys
import json
import yaml
import pickle
import joblib

import numpy as np

from pathlib import Path
from typing import Any

from box import ConfigBox
from box.exceptions import BoxValueError
from ensure import ensure_annotations
from scipy.sparse import csr_matrix, save_npz, load_npz
from src.logger.custom_logger import logger
from src.exception.custom_exception import CustomException

@ensure_annotations
def read_yaml(file_path: Path) -> ConfigBox:
    try:
        if not file_path.exists():
            raise FileNotFoundError(f"YAML file not found: {file_path}")
        
        logger.info(f"Reading YAML file: {file_path}")
            
        with open (file_path, "r", encoding="utf-8") as yaml_file:
            content = yaml.safe_load(yaml_file)

        if content is None:
            raise ValueError("YAML file is empty")

        logger.info(f"YAML file: {file_path} loaded successfully.")
        return ConfigBox(content)

    except BoxValueError:
        raise ValueError("Invalid YAML Format.")
    


    
def write_yaml(file_path: Path, content: Any) -> None:
    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)

        with open(file_path, "w", encoding="utf-8") as file:
            yaml.safe_dump(content, file, sort_keys=False)

        logger.info(f"YAML file saved at: {file_path}")

    except Exception as e:
        raise CustomException(e, sys)




def create_directories(path_to_directories: list[Path], verbose: bool = True) -> None:
    try:
        for path in path_to_directories:
            path.mkdir(parents=True, exist_ok=True)

            if verbose:
                logger.info(f"Created directory at: {path}")

    except Exception as e:
        raise CustomException(e, sys)
    


#@ensure_annotations
def save_json(path: Path, data: dict) -> None:
    """Saves dictionary data to a JSON file."""

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=4)

        logger.info(f"JSON file saved at: {path}")

    except Exception as e:
        raise CustomException(e, sys)

#@ensure_annotations
def load_json(path: Path) -> ConfigBox:
    """Loads JSON file data as class attributes."""

    try:
        with open(path, "r") as f:
            content = json.load(f)

        logger.info(f"JSON file loaded successfully from: {path}")
        return ConfigBox(content)
    
    except Exception as e:
        raise CustomException(e, sys)


#@ensure_annotations
def save_bin(data: Any, path: Path) -> None:
    """Saves binary file using joblib."""

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(value=data, filename=path)

        logger.info(f"Binary file saved at: {path}")

    except Exception as e:
        raise CustomException(e, sys)


#@ensure_annotations
def load_bin(path: Path) -> Any:
    """Loads binary data using joblib."""

    try:
        data = joblib.load(path)

        logger.info(f"Binary file loaded from: {path}")
        return data
    
    except Exception as e:
        raise CustomException(e, sys)

# @ensure_annotations
def save_numpy_array_data(file_path: Path, array: np.ndarray) -> None:
    """Saves numpy array data to a file."""

    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            np.save(file_obj, array)

        logger.info(f"Numpy array saved successfully at: {file_path}")

    except Exception as e:
        raise CustomException(e, sys)

#@ensure_annotations
def load_numpy_array_data(file_path: Path) -> np.ndarray:
    """Loads numpy array data from a file."""

    try:
        with open(file_path, "rb") as file_obj:
            return np.load(file_obj)
        
    except Exception as e:
        raise CustomException(e, sys)

#@ensure_annotations obj: Any(accepting generic object)
def save_object(file_path: Path, obj: Any) -> None:
    """Pickles a python object using standard serialization."""

    try:
        logger.info(f"Saving pickle object to: {file_path}")
        file_path.parent.mkdir(parents=True, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)

#@ensure_annotations
def load_object(file_path: Path) -> Any:
    """Loads a pickled object from a file path."""

    try:
        if not file_path.exists():
            raise FileNotFoundError(f"The file: {file_path} does not exist.")
        
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)
        
    except Exception as e:
        raise CustomException(e, sys)

def save_sparse_matrix(file_path: Path, matrix: csr_matrix) -> None:
    """Save a scipy sparse matrix."""

    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)

        save_npz(file_path, matrix)

        logger.info(f"Sparse matrix saved successfully at: {file_path}")

    except Exception as e:
        raise CustomException(e, sys)


def load_sparse_matrix(file_path: Path) -> csr_matrix:
    """Load a scipy sparse matrix."""

    try:
        matrix = load_npz(file_path)

        logger.info(f"Sparse matrix loaded successfully from: {file_path}")

        return matrix

    except Exception as e:
        raise CustomException(e, sys)