import os
from pathlib import Path
import logging

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s]: %(message)s'
)

list_of_files = [

    #------------ app — fastapi/flask api + dashboard ----------------

    "app/__init__.py",                      # app factory — create_app()
    "app/app.py",                           # mounts routes, starts server, wires config
    "app/routes/prediction.py",             # POST /predict — text in, category+confidence out
    "app/routes/health.py",                 # GET /health — liveness/readiness check for CodeDeploy validation
    "app/routes/metrics.py",                # GET /metrics — Prometheus scrape endpoint
    "app/templates/index.html",             # text input box + predicted category + confidence bar chart
    "app/static/style.css",                 # minimal dashboard styling

    #------------ artifacts — dvc tracked, pushed to s3 --------

    "artifacts/models/",                    # baseline_tfidf_logreg.pkl · distilbert_finetuned/ · label_encoder.pkl
    "artifacts/vectorizers/",               # fitted tfidf_vectorizer.pkl (classical path only)
    "artifacts/tokenizers/",                # fitted HF tokenizer files (transformer path only)
    "artifacts/metrics/",                   # metrics.json per model, confusion matrices as JSON
    "artifacts/reports/",                   # model_comparison.md rendered output, eval reports
    "artifacts/plots/",                     # confusion_matrix.png · per-class F1 bar chart · class distribution plot

    #------------ config — single source of truth --------------

    "config/config.yaml",                   # text/label column names · min/max text length · expected class labels
    "config/schema.yaml",                   # ALL paths · hyperparams (lr, epochs, max_len, batch_size) · AWS settings
    "config/logging.yaml",                  # log format · rotating file handler settings · log level per module

    #------------ data — dvc tracked --------------------------

    "data/raw/",                            # untouched downloaded/scraped news corpus
    "data/interim/",                        # partially cleaned data — post data_validation, pre split
    "data/processed/",                      # final train.csv / val.csv / test.csv, post transformation
    "data/external/",                       # any third-party reference data (e.g. stopword lists, label taxonomy)

    #------------ ci/cd — github actions + aws codepipeline------------

    ".github/workflows/ci.yml",             # lint + unit tests on every push — blocks deploy on failure
    ".github/workflows/cd.yml",             # docker build → push ECR → trigger CodePipeline → deploy EC2

    #-------------------------- DOCS ------------------------

    "docs/architecture.png",                # data → baseline → transformer → serving diagram
    "docs/api_reference.md",                # /predict request/response schema, error codes
    "docs/model_comparison.md",             # baseline vs transformer: accuracy, F1-macro, latency, model size table
    "docs/deployment.md",                   # how CI/CD pipeline works, how to deploy manually if needed
    "docs/monitoring.md",                   # what drift_detector/metrics_logger track and how to read dashboards

    # ----------------------- notebooks — eda and experiments ----------------------

    "notebooks/01_eda.ipynb",                       # class distribution · text length dist · vocabulary size · imbalance check
    "notebooks/02_preprocessing.ipynb",             # cleaning strategy experiments — what to strip, what to keep
    "notebooks/03_feature_engineering.ipynb",       # TF-IDF param sweeps (ngram range, max_features) vs tokenizer max_len choice
    "notebooks/04_classical_models.ipynb",          # TF-IDF + LogReg/SVM baseline benchmark
    "notebooks/05_transformer_training.ipynb",      # DistilBERT fine-tune experiments
    "notebooks/06_model_comparison.ipynb",          # baseline vs transformer side-by-side, feeds docs/model_comparison.md

    # -------------------------- pipeline — top-level orchestrators ------------------

    "pipeline/training_pipeline.py",                # calls src/stages in order — single entry point for full training run
    "pipeline/prediction_pipeline.py",               # loads artifacts → batch classify new text CSV
    "pipeline/retraining_pipeline.py",              # triggered by Lambda on new labeled data — reruns training_pipeline, compares to prod model before promoting

    # ----------------------------------- src — main package (all logic lives here) -----------------------------------
    # ----------------------------------- COMPONENTS: one class per responsibility — the core logic -----------------------

    "src/__init__.py",
    "src/components/__init__.py",

    "src/components/data_ingestion.py",             # class DataIngestion — pull corpus from S3/HF hub → save to data/raw
    "src/components/data_validation.py",            # class DataValidation — check label set matches schema · null % · text length bounds → data/interim
    "src/components/data_transformation.py",        # class DataTransformation — ORCHESTRATOR ONLY: stratified split, calls TextPreprocessor + FeatureBuilder in sequence, saves data/processed. Owns split logic exclusively — no cleaning/vectorization code lives here.
    # leave the test set as it is till the final evaluation and use train set for train, test and val split


    "src/components/text_preprocessor.py",          # class TextPreprocessor — CLEANING ONLY. clean_for_classical() (lowercase, strip HTML/URL, remove stopwords, lemmatize) vs clean_for_transformer() (minimal cleaning, no stopword/lemma removal — tokenizer needs natural structure). No vectorization, no splitting.
    "src/components/feature_builder.py",            # class FeatureBuilder — CLEAN TEXT -> NUMBERS ONLY. build_classical_features() fits TF-IDF on train only, transforms val/test. build_transformer_features() tokenizes via HF tokenizer into input_ids/attention_mask. encode_labels() shared LabelEncoder for both paths. No cleaning, no splitting.

    "src/components/traditional_model_trainer.py",  # class TraditionalModelTrainer — trains TF-IDF + LogisticRegression/LinearSVC baseline · logs params/metrics to MLflow
    "src/components/transformer_trainer.py",        # class TransformerTrainer — fine-tunes DistilBERT via HF Trainer/Accelerate · logs params/metrics to MLflow · saves checkpoint

    "src/components/model_evaluator.py",            # class ModelEvaluator — per-class precision/recall/F1, confusion matrix, ROC-AUC per class, for EACH model independently
    "src/components/model_selector.py",             # class ModelSelector — explicit decision logic: compares evaluator outputs for both models against config thresholds, picks winner, writes reasoning to artifacts/reports/
    "src/components/model_exporter.py",             # class ModelExporter — packages winning model + its preprocessor/vectorizer/tokenizer into a single deployable artifact bundle, pushes to S3/artifact store

    # ------------------------------- CONFIG: reads config.yaml and schema.yaml into typed Python dataclasses ----------------------

    "src/config/__init__.py",
    "src/config/configuration.py",                  # class ConfigurationManager — read_data_config() · read_model_config() · read_aws_config() · read_logging_config()

    # ------------------------------------ UTILS: shared helpers — imported by components, no ML logic here -------------------------

    "src/utils/__init__.py",
    "src/utils/common.py",                          # read_yaml() · save_json() · load_json() · save_object() · load_object() · create_directories()
    "src/utils/text_utils.py",                      # low-level string helpers used by text_preprocessor.py (regex patterns, HTML stripping primitives)
    "src/utils/metrics.py",                         # shared metric computation functions (F1, precision/recall) used by model_evaluator.py
    "src/utils/visualization.py",                   # confusion matrix plotting, class distribution bar charts — used by model_evaluator.py and notebooks
    "src/utils/model_utils.py",                     # save/load model checkpoints (both sklearn .pkl and HF checkpoint dirs) with consistent interface
    "src/utils/storage_utils.py",                   # upload_to_s3() · download_from_s3() · check_s3_key_exists() — replaces old s3_utils.py naming

    # ------------------------EXCEPTION: custom exception class with file name + line number in message -----------------------------

    "src/exception/__init__.py",
    "src/exception/custom_exception.py",            # class NewsClassifierException(Exception) — wraps sys.exc_info() with file+line context

    # ------------------ LOGGING: structured logging — every component imports from here -----------------------

    "src/logger/__init__.py",
    "src/logger/logger.py",                         # setup_logger() — reads config/logging.yaml, rotating file handler + stdout

    # ------------------------ STAGES: stage-level runners — called by top-level pipeline/ orchestrators ---------------------------

    "src/stages/stage_01_data_ingestion.py",        # instantiates DataIngestion · calls initiate() · logs start/end
    "src/stages/stage_02_data_validation.py",       # instantiates DataValidation · raises on schema failure
    "src/stages/stage_03_data_transformation.py",   # instantiates DataTransformation (with TextPreprocessor + FeatureBuilder injected) · saves train/val/test artifacts
    "src/stages/stage_04_model_training.py",        # instantiates TraditionalModelTrainer + TransformerTrainer · trains both · logs to MLflow
    "src/stages/stage_05_model_evaluation.py",      # instantiates ModelEvaluator + ModelSelector · saves metrics.json + selection reasoning
    "src/stages/stage_06_model_export.py",          # instantiates ModelExporter · packages winning model bundle · pushes to artifact store

    # ---------------------------------------ENTITY: typed config/artifact dataclasses --------------------------

    "src/entity/config_entity.py",                  # @dataclass DataIngestionConfig · DataTransformationConfig · ModelTrainerConfig · etc.
    "src/entity/artifact_entity.py",                # @dataclass DataIngestionArtifact · DataTransformationArtifact · ModelTrainerArtifact · etc.
    "src/entity/prediction_entity.py",              # @dataclass PredictionInput · PredictionOutput — request/response contracts used by predictor.py and app/routes/prediction.py

    "src/constants/constants.py",                   # ARTIFACT_DIR · MODEL_NAME · MAX_LEN · CLASS_LABELS · DEFAULT thresholds

    # --------------------------------MONITORING: model health after deployment ---------------------

    "src/monitoring/drift_detector.py",             # class DriftDetector — compares live text vocabulary/class distribution to training distribution · flags drift · logs to MLflow
    "src/monitoring/metrics_logger.py",             # class MetricsLogger — Prometheus counters: prediction volume per class, error rate
    "src/monitoring/latency_monitor.py",            # class LatencyMonitor — tracks p50/p95/p99 inference latency, alerts if threshold breached
    "src/monitoring/data_quality_monitor.py",       # class DataQualityMonitor — flags incoming requests with out-of-bounds text length, empty strings, unexpected encoding

    "src/predictor.py",                             # class Predictor — loads model+vectorizer/tokenizer bundle once at startup · predict(text) → category + confidence + top-3 probabilities

    # ------------------------- tests — pytest --------------------------------

    "tests/test_data_validation.py",                # schema check · label set validation · text length bounds
    "tests/test_text_preprocessor.py",              # clean_for_classical() vs clean_for_transformer() output correctness, edge cases (empty string, HTML, URLs)
    "tests/test_feature_builder.py",                # TF-IDF fit-on-train-only (no leakage) · tokenizer output shape · label encoding consistency
    "tests/test_model_training.py",                 # smoke test: both trainers run on tiny sample data without error, produce valid artifact
    "tests/test_predictor.py",                      # output dict shape · confidence in [0,1] · category in valid label set
    "tests/test_api.py",                            # /predict 200 · /health 200 · empty text 400 · text too long 422
    "tests/test_pipeline.py",                       # end-to-end smoke test: training_pipeline.py runs on tiny sample dataset without error

    # ------------------------------------------------ aws codedeploy hooks ----------------------

    "scripts/build_container.sh",                   # docker build with correct tags
    "scripts/pull_images.sh",                       # ECR login + docker pull latest
    "scripts/start_container.sh",                   # docker run --restart always -p 8000:8000
    "scripts/stop_container.sh",                    # docker stop + rm old container
    "scripts/validate_services.sh",                 # curl /health × 10 retries — auto-rollback on failure

    # --------------------------- deployment — codedeploy + prod compose ---------------------------

    "deployment/appspec.yml",                       # lifecycle hooks — BeforeInstall · AfterInstall · Start · Validate
    "deployment/buildspec.yml",                     # AWS CodeBuild — build Docker → push ECR → write imagedefinitions.json
    "deployment/docker-compose.prod.yml",           # production compose: app + prometheus, no dev-only services

    # --------------------------- aws lambda — auto-retrain trigger ---------------------------------

    "aws/s3_trigger_lambda.py",                     # fires when new labeled batch lands in S3 → triggers retraining_pipeline.py via CodePipeline
    "aws/iam_policy.json",                          # least-privilege IAM for EC2 + CodePipeline + S3 + ECR + Lambda

    # --------------------------- mlflow tracking ---------------------------

    "mlruns/",                                      # local MLflow tracking store (or configured to point at DagsHub remote)

    "dvc.yaml",                                     # pipeline DAG — stage_01 → 02 → 03 → 04 → 05 → 06 with deps and outs
    "params.yaml",                                  # DVC-tracked hyperparams — change here, dvc repro, compare in MLflow

    "Dockerfile",                                   # multi-stage build: install deps, copy src, run app.py via uvicorn
    "docker-compose.yml",                           # app + mlflow UI + prometheus — local dev stack in one command
    ".flake8",                                      # lint rules
    ".env.example",                                 # AWS_ACCESS_KEY · AWS_SECRET_KEY · MLFLOW_TRACKING_URI · HF_TOKEN placeholders
    "requirements.txt",                             # production deps — pandas · scikit-learn · transformers · torch · mlflow · fastapi · boto3 · prometheus-client
    "requirements-dev.txt",                         # dev/test deps — pytest · flake8 · black · isort · mypy · jupyter · dvc[s3] · pre-commit
    "setup.py",                                     # pip install -e . — makes all src/ sub-packages importable anywhere
    "README.md",                                    # architecture diagram · baseline vs transformer results table · live URL · quickstart · API reference
   # ".gitignore",
    "main.py",                                      # entry point — runs training_pipeline.py end to end

    # NOTE: all inline comments above written to clarify single-responsibility
    # boundaries per file, per locked component-boundary decisions.
]

for item in list_of_files:

    is_directory = item.endswith("/")
    path = Path(item)

    # Skip if it already exists (file or directory)
    if path.exists():
        logging.info(f"Already exists: {path}")
        continue

    if is_directory:
        path.mkdir(parents=True, exist_ok=True)
        logging.info(f"Directory created: {path}")

    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
        logging.info(f"File created: {path}")