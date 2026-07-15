from pathlib import Path

from src.constants.constants import *
from src.utils.common import read_yaml, create_directories

from src.entity.config_entity import (DataIngestionConfig, 
                                      DataValidationConfig, 
                                      TextPreprocessorConfig,
                                      TFIDFConfig,
                                      TransformerConfig, 
                                      FeatureBuilderConfig, 
                                      DataTransformationConfig,
                                      TraditionalModelTrainerConfig,
                                      RandomSearchParams,
                                      LogisticRegressionParams,
                                      MultinomialNaiveBayesParams,
                                      LinearSVMParams,
                                      
                                      LightGBMParams,
                                      ModelTrainerParams,
                                      ModelEvaluationConfig)

class ConfigurationManager:
    def __init__(self,
                 config_file_path = CONFIG_FILE_PATH,
                 schema_file_path = SCHEMA_FILE_PATH,
                 params_file_path = PARAMS_FILE_PATH):
        
        self.config = read_yaml(config_file_path)
        self.schema = read_yaml(schema_file_path)
        self.params = read_yaml(params_file_path)

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        data_ingestion_config = DataIngestionConfig(
            dataset_name=config.dataset_name,
            dataset_download_dir=Path(config.dataset_download_dir),
            train_dataset=config.train_dataset,
            test_dataset=config.test_dataset
        )

        return data_ingestion_config
    
    def get_data_validation_config(self)-> DataValidationConfig:
        config = self.config.data_validation
        schema = self.schema.COLUMNS
        target_column = self.schema.TARGET_COLUMN.target

        text_columns=list(schema.keys())
        text_columns.remove(target_column)

        expected_labels={1, 2, 3, 4}

        validation_root = Path(config.root_dir)
        create_directories([validation_root])

        data_validation_config = DataValidationConfig(
            root_dir=validation_root,
            status_file_path=Path(config.STATUS_FILE),
            dataset_path=Path(config.dataset_path),
            all_schema=schema,
            target_column=target_column,
            text_columns=text_columns,
            expected_labels=expected_labels
        )

        return data_validation_config
    

    def get_text_preprocessor_config(self) -> TextPreprocessorConfig:
        config = self.config.text_preprocessor
        params = self.params


        text_preprocessor_root = Path(config.root_dir)
        create_directories([text_preprocessor_root])

        text_preprocessor_config = TextPreprocessorConfig(
            root_dir=text_preprocessor_root,
            remove_stopwords=params.text_preprocessor.remove_stopwords,
            lemmatize=params.text_preprocessor.lemmatize,
            min_token_length=params.text_preprocessor.min_token_length

        )

        return text_preprocessor_config



    def get_feature_builder_config(self) -> FeatureBuilderConfig:
        config = self.config.feature_builder
        params = self.params

        feature_builder_root = Path(config.root_dir)
        create_directories([feature_builder_root, Path(config.vectorizer_path).parent])

        feature_builder_config = FeatureBuilderConfig(
                root_dir=feature_builder_root,
                vectorizer_path=Path(config.vectorizer_path),

                tfidf=TFIDFConfig(
                    max_features=params.feature_builder.tfidf.max_features,
                    ngram_range=tuple(params.feature_builder.tfidf.ngram_range),
                    min_df=params.feature_builder.tfidf.min_df,
                    max_df=params.feature_builder.tfidf.max_df,
                ),

                transformer=TransformerConfig(
                    model_name=params.feature_builder.transformer.model_name,
                    max_length=params.feature_builder.transformer.max_length,
                )
            )

        return feature_builder_config
    


    def get_data_transformation_config(self) -> DataTransformationConfig:
        config = self.config.data_transformation
        params = self.params.data_transformation
        schema = self.schema.COLUMNS

        target_column = self.schema.TARGET_COLUMN.target

        text_columns=list(schema.keys())
        text_columns.remove(target_column)

        create_directories([
                Path(config.root_dir),
                Path(config.processed_train_data_path).parent,
                Path(config.processed_validation_data_path).parent,
                Path(config.processed_test_data_path).parent,
            ])

        data_transformation_config = DataTransformationConfig(
            root_dir=Path(config.root_dir),

            train_path=Path(config.train_path),
            target_column=target_column,
            text_columns=text_columns,

            test_size=params.test_size,
            validation_size=params.validation_size,
            random_state=params.random_state,

            processed_train_data_path=Path(config.processed_train_data_path),
            processed_validation_data_path=Path(config.processed_validation_data_path),
            processed_test_data_path=Path(config.processed_test_data_path),

            processed_train_target_path=Path(config.processed_train_target_path),
            processed_validation_target_path=Path(config.processed_validation_target_path),
            processed_test_target_path=Path(config.processed_test_target_path),

        )

        return data_transformation_config
    

    def get_model_trainer_config(self) -> TraditionalModelTrainerConfig:
        config = self.config.traditional_model_trainer

        model_root = Path(config.root_dir)
        # trained_models_dir = Path(config.trained_models_dir)
        search_results_dir = Path(config.search_results_dir)

        create_directories([
            model_root,
            # trained_models_dir,
            search_results_dir
        ])

        model_trainer_config = TraditionalModelTrainerConfig(
            root_dir=model_root,
            # trained_models_dir=trained_models_dir,
            best_model_path=Path(config.best_model_path),
            best_params_path=Path(config.best_params_path),
            model_report_path=Path(config.model_report_path),
            search_results_dir=search_results_dir

        )

        return model_trainer_config
    
    # design data transformation and adjust Text_preprocessor and feature_builder similar to model_trainer_params

    def get_model_trainer_params(self) -> ModelTrainerParams:
        params = self.params

        random_search = RandomSearchParams(
            n_iter=params.random_search.n_iter,
            cv=params.random_search.cv,
            scoring=params.random_search.scoring,
            random_state=params.random_search.random_state,
            n_jobs=params.random_search.n_jobs,
            verbose=params.random_search.verbose
            
        )

        logstic_regression = LogisticRegressionParams(
            C=params.logistic_regression.C,
            penalty=params.logistic_regression.penalty,
            solver=params.logistic_regression.solver,
            max_iter=params.logistic_regression.max_iter,
            class_weight=params.logistic_regression.class_weight

        )

        multinomial_naive_bayes = MultinomialNaiveBayesParams(
            alpha=params.multinomial_naive_bayes.alpha,
            fit_prior=params.multinomial_naive_bayes.fit_prior

        )

        linear_SVM = LinearSVMParams(
            C=params.linear_svm.C,
            loss=params.linear_svm.loss,
            max_iter=params.linear_svm.max_iter,
            class_weight=params.linear_svm.class_weight
        )

        lightgbm = LightGBMParams(
            n_estimators=params.lightgbm.n_estimators,
            learning_rate=params.lightgbm.learning_rate,
            num_leaves=params.lightgbm.num_leaves,
            max_depth=params.lightgbm.max_depth,
            feature_fraction=params.lightgbm.feature_fraction,
            bagging_fraction=params.lightgbm.bagging_fraction,
            min_child_samples=params.lightgbm.min_child_samples,
            lambda_l1=params.lightgbm.lambda_l1,
            lambda_l2=params.lightgbm.lambda_l2

        )

        model_trainer_params = ModelTrainerParams(
            random_search=random_search,

            logistic_regression=logstic_regression,
            multinomial_naive_bayes=multinomial_naive_bayes,
            linear_svm=linear_SVM,
            lightgbm=lightgbm

        )

        return model_trainer_params

        
    #------------Model Evaluation-------------

    def get_model_evaluation_config(self) -> ModelEvaluationConfig:

        config = self.config.model_evaluation

        evaluation_root = Path(config.root_dir)
        create_directories([evaluation_root])

        model_evaluation_config = ModelEvaluationConfig(
            root_dir=evaluation_root,
            evaluation_report_path=Path(config.evaluation_report_path),
            classification_report_path=Path(config.classification_report_path),
            confusion_matrix_path=Path(config.confusion_matrix_path),
            evaluation_summary_path=Path(config.evaluation_summary_path),
            threshold_analysis_path=Path(config.threshold_analysis_path)
        )

        return model_evaluation_config
    
# deployment config remains