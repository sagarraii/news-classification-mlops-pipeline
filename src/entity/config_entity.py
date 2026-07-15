import numpy as np
from scipy.sparse import csr_matrix
from pathlib import Path
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    dataset_name: str
    dataset_download_dir: Path
    train_dataset: str
    test_dataset: str


@dataclass
class DataValidationConfig:
    root_dir: Path
    status_file_path: Path
    dataset_path: Path
    all_schema: dict
    target_column: str
    text_columns: list[str]
    expected_labels: set[int]


@dataclass(frozen=True)
class TextPreprocessorConfig:
    root_dir: Path
    remove_stopwords: bool
    lemmatize: bool
    min_token_length: int



@dataclass(frozen=True)
class TFIDFConfig:
    max_features: int
    ngram_range: tuple
    min_df: int
    max_df: float


@dataclass(frozen=True)
class TransformerConfig:
    model_name: str
    max_length: int


@dataclass(frozen=True)
class FeatureBuilderConfig:
    root_dir: Path

    vectorizer_path: Path

    tfidf: TFIDFConfig
    transformer: TransformerConfig


@dataclass(frozen=True)
class DataTransformationConfig:
    root_dir: Path

    train_path: Path

    target_column: str
    text_columns: list[str]

    test_size:float
    validation_size: float
    random_state: int

    processed_train_data_path: Path
    processed_validation_data_path: Path
    processed_test_data_path: Path

    processed_train_target_path: Path
    processed_validation_target_path: Path
    processed_test_target_path: Path

@dataclass
class TrainingData:
    X_train: csr_matrix
    y_train: np.ndarray
    X_valid: csr_matrix
    y_valid: np.ndarray

@dataclass
class TraditionalModelTrainerConfig:
    root_dir: Path
    # trained_models_dir: Path
    best_model_path: Path
    best_params_path: Path
    model_report_path: Path
    search_results_dir: Path


@dataclass
class RandomSearchParams:
    n_iter: int
    cv: int
    scoring: str
    random_state: int
    n_jobs: int
    verbose: int

@dataclass(frozen=True)
class LogisticRegressionParams:
    C: list[float]
    penalty: list[str]
    solver: list[str]
    max_iter: list[int]
    class_weight: list[str | None]
    

@dataclass(frozen=True)
class MultinomialNaiveBayesParams:
    alpha: list[float]
    fit_prior: list[bool]

@dataclass(frozen=True)
class LinearSVMParams:
    C: list[float]
    loss: list[str]
    max_iter: list[int]
    class_weight: list[str | None]

@dataclass(frozen=True)
class LightGBMParams:
    n_estimators: list[int]
    learning_rate: list[float]
    num_leaves: list[int]
    max_depth: list[int]
    feature_fraction: list[float]
    bagging_fraction: list[float]
    min_child_samples: list[int]
    lambda_l1: list[float]
    lambda_l2: list[float]



@dataclass(frozen=True)
class ModelTrainerParams:
    random_search: RandomSearchParams

    logistic_regression: LogisticRegressionParams
    multinomial_naive_bayes: MultinomialNaiveBayesParams
    linear_svm: LinearSVMParams
    lightgbm: LightGBMParams

@dataclass
class TestingData:
    X_test: csr_matrix
    y_test: np.ndarray


@dataclass
class ModelEvaluationConfig:
    root_dir: Path
    evaluation_report_path: Path
    classification_report_path: Path
    confusion_matrix_path: Path
    evaluation_summary_path: Path
    threshold_analysis_path: Path

