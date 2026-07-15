import sys
import pandas as pd

from datetime import datetime
from pathlib import Path

from src.logger.custom_logger import logger
from src.exception.custom_exception import CustomException

from src.entity.config_entity import DataValidationConfig



class DataValidation:
    def __init__(self, config: DataValidationConfig) -> None:
        self.config = config


    # -------------------File Existence ------------------------


    @staticmethod
    def check_file_existence(dataset_path: Path) -> None:
        try:
            if not dataset_path.exists():

                logger.error(f"Dataset file not found at: '{dataset_path}'.")
                raise FileNotFoundError(f"The dataset file was not found at: {dataset_path}")
            
            logger.info(f"Dataset file found: '{dataset_path}'.")
            
        except Exception as e:
            raise CustomException(e, sys)
        

    # ---------------------Readig data fron local source----------------------


    @staticmethod
    def read_data(dataset_path: Path) -> pd.DataFrame:
        try:
            logger.info(f"Loading dataset from '{dataset_path}'.")

            df = pd.read_csv(dataset_path)

            logger.info(f"Dataset loaded successfully with {len(df)} records.")          
            return df
        
        except Exception as e:
            raise CustomException (e, sys)


    # ------------------column validation -----------------------------


    def column_validation(self, dataframe: pd.DataFrame) -> tuple[bool, list]:
        try:
            schema_columns = list(self.config.all_schema.keys())
            dataframe_columns = list(dataframe.columns)
            missing_columns = [col for col in schema_columns if col not in dataframe_columns]

            if missing_columns:
                logger.warning(f"Schema validation failed. Missing columns: {missing_columns}.")
                return False, missing_columns

            logger.info("Schema validation passed. All required columns are present.")
            return True, []
        
        except Exception as e:
            raise CustomException(e, sys)


    # -------------------------------checking unexpected columns -----------


    def validate_unexpected_columns(self, dataframe: pd.DataFrame) -> tuple[bool, list]:
        try:
            schema_columns = list(self.config.all_schema.keys())
            dataframe_columns = list(dataframe.columns)

            unexpected_columns = [col for col in dataframe_columns if col not in schema_columns]

            if unexpected_columns:
                logger.warning(f"Unexpected columns detected: {unexpected_columns}.")
                return False, unexpected_columns

            logger.info("No unexpected columns found.")
            return True, []

        except Exception as e:
            raise CustomException(e, sys)
    
        
    #-------------------------- data type validation from schema --------------------------------


    def check_dtype(self, dataframe: pd.DataFrame) -> tuple[bool, dict]:
        try:
            mismatches = {}

            for column, expected_dtype in self.config.all_schema.items():
                if column not in dataframe.columns:
                    continue

                actual_dtype = str(dataframe[column].dtype)

                if actual_dtype != expected_dtype:
                    mismatches[column] = {"expected": expected_dtype, "found": actual_dtype}

                    logger.warning(
                            f"Data type mismatch for column '{column}'. "
                            f"Expected '{expected_dtype}', found '{actual_dtype}'."
                        )

            return len(mismatches) == 0, mismatches
        
        except Exception as e:
            raise CustomException(e, sys)
    

    # --------------------------------Target column validation ------------------------
    

    def validate_target_column_exists(self, dataframe: pd.DataFrame) -> bool:
       
        try:
            target_column = self.config.target_column

            if target_column not in dataframe.columns:
                logger.warning(f"Target column '{target_column}' was not found.")
                return False
            
            logger.info(f"Target column '{target_column}' found.")
            return True
        
        except Exception as e:
            raise CustomException(e, sys)
        

    # ------------------ Target inputs Validation ------------------------------------


    def validate_target_inputs(self, dataframe: pd.DataFrame) -> tuple[bool, dict]:
        try:
            actual = set(dataframe[self.config.target_column].unique())
            expected = self.config.expected_labels
            
            invalid = actual - expected
            missing = expected - actual
            
            if invalid or missing:
                mismatches = {
                    "expected": sorted(expected),
                    "found": sorted(actual),
                    "invalid": sorted(invalid),
                    "missing": sorted(missing),
                }

                logger.warning(
                        "Target label validation failed. "
                        f"Unexpected labels: {sorted(invalid)}, "
                        f"Missing labels: {sorted(missing)}."
                    )
            else:
                mismatches = {}
                logger.info("Target labels validated successfully.")

            return len(mismatches) == 0, mismatches
        
        except Exception as e:
            raise CustomException(e, sys)
            

    #---------------------- checking missing values (if any) --------------------------


    def check_missing_values(self, dataframe: pd.DataFrame) -> tuple[bool, dict]:
        try:
            missing = dataframe.isnull().sum()
            columns_with_nulls = missing[missing > 0].to_dict()

            if columns_with_nulls:
                logger.warning(f"Missing values detected in columns: {columns_with_nulls}.")
                return False, columns_with_nulls

            logger.info("No missing values found.")
            return True, {}
        
        except Exception as e:
            raise CustomException(e, sys)
        

    # -----------------------validation White-space if any in column ["title", "description"]------------------------


    def validate_whitespace_text(self, dataframe: pd.DataFrame) -> tuple[bool, dict]:
        try:
            issues = {}

            for col in self.config.text_columns:
                empty = dataframe[col].fillna("").astype(str).str.strip().eq("").sum()

                if empty > 0:
                    issues[col] = int(empty)
                    logger.warning(f"Whitespace-only values detected in column '{col}': {empty} rows.")

            return len(issues) == 0, issues

        except Exception as e:
            raise CustomException(e, sys)
        

    #----------------- distribution of target inputs ------------------------------


    def validate_class_distribution(self, dataframe: pd.DataFrame) -> tuple[bool, dict]:
        try:
            distribution = (dataframe[self.config.target_column].value_counts().sort_index())
            logger.info(f"Target variable distribution: {distribution.to_dict()}")

            return True, distribution.to_dict()
        
        except Exception as e:
            raise CustomException(e, sys)


    # -----------------------counting total number of rows and columns-------------------------


    def get_dataset_shape(self, dataframe: pd.DataFrame) -> tuple[int, int]:
        try:
            total_rows, total_cols = dataframe.shape

            logger.info(f"Dataset shape: {total_rows} rows × {total_cols} columns.")
            return total_rows, total_cols

        except Exception as e:
            raise CustomException(e, sys)


    #--------------------- checking for duplicate if any in rows-------------------------------


    def check_duplicates(self, dataframe: pd.DataFrame) -> tuple[bool, int]:
        try:
            duplicate_count = int(dataframe.duplicated().sum())

            if duplicate_count > 0:
                logger.warning(f"Duplicate rows detected: {duplicate_count}.")
                return False, duplicate_count

            logger.info("No duplicate rows detected.")
            return True, 0
        
        except Exception as e:
            raise CustomException(e, sys)
        

    # -------------------- Defining final report of Data Validation (status.txt)--------------------------------    
        

    def write_validation_report(
        self,
        dataframe: pd.DataFrame,
        overall_status: bool,
        missing_columns: list,
        unexpected_columns: list,
        dtype_mismatches: dict,
        target_exists: bool,
        target_label_issues: dict,
        missing_values: dict,
        whitespace_issues: dict,
        duplicate_count: int,
        class_distribution: dict,
    ):
        """Writes the final data validation report."""

        try:
            rows, cols = dataframe.shape
            numeric_cols = dataframe.select_dtypes(include="number").shape[1]

            report = [
                "=" * 70,
                "DATA VALIDATION REPORT",
                "=" * 70,
                f"Generated At           : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                f"Dataset                : {self.config.dataset_path}",
                "",
                "-" * 70,
                "DATASET OVERVIEW",
                "-" * 70,
                f"Rows                   : {rows}",
                f"Columns                : {cols}",
                f"Numeric Columns        : {numeric_cols}",
                f"Categorical Columns    : {cols - numeric_cols}",
                "",
                "-" * 70,
                "VALIDATION RESULTS",
                "-" * 70,
            ]

            checks = [
                ("Column Presence", not missing_columns),
                ("Unexpected Columns", not unexpected_columns),
                ("Data Types", not dtype_mismatches),
                ("Target Column Exists", target_exists),
                ("Target Labels", not target_label_issues),
                ("Missing Values", not missing_values),
                ("Whitespace Text", not whitespace_issues),
                ("Duplicate Rows", duplicate_count == 0),
            ]

            for name, status in checks:
                report.append(f"[{'PASS' if status else 'FAIL'}] {name}")

            report.append("\nDETAILS")
            report.append("-" * 70)

            if missing_columns:
                report.append(f"Missing Columns        : {missing_columns}")

            if unexpected_columns:
                report.append(f"Unexpected Columns     : {unexpected_columns}")

            if dtype_mismatches:
                report.append("\nDatatype Mismatches")
                for col, info in dtype_mismatches.items():
                    report.append(
                        f"  - {col}: expected={info['expected']} found={info['found']}"
                    )

            if not target_exists:
                report.append(
                    f"Target Column Missing  : {self.config.target_column}"
                )

            if target_label_issues:
                report.append(
                    f"Target Label Issues    : {target_label_issues}"
                )

            if missing_values:
                report.append("\nMissing Values")
                for col, count in missing_values.items():
                    report.append(f"  - {col}: {count}")

            if whitespace_issues:
                report.append("\nWhitespace-only Text")
                for col, count in whitespace_issues.items():
                    report.append(f"  - {col}: {count}")

            if duplicate_count:
                report.append(f"\nDuplicate Rows         : {duplicate_count}")

            report.append("\nClass Distribution")
            for label, count in class_distribution.items():
                report.append(f"  Label {label}: {count}")

            report.extend([
                "",
                "-" * 70,
                f"FINAL STATUS : {'PASSED' if overall_status else 'FAILED'}",
                "=" * 70,
            ])

            with open(self.config.status_file_path, "w") as file:
                file.write("\n".join(report))

            logger.info(f"Validation report saved to '{self.config.status_file_path}'.")

        except Exception as e:
            raise CustomException(e, sys)
        

    # ----------------- Initating Data Validation -----------------------


    def initiate_data_validation(self, dataset_path: Path = None) -> bool:
        """Runs all validation checks and generates the validation report."""

        try:
            logger.info("Starting data validation pipeline.")

            path = dataset_path or self.config.dataset_path

            # Check file existence
            self.check_file_existence(path)

            # Read dataset only once
            dataframe = self.read_data(path)

            # Validation checks
            col_status, missing_columns = self.column_validation(dataframe)
            unexpected_status, unexpected_cols = self.validate_unexpected_columns(dataframe)
            dtype_status, dtype_mismatches = self.check_dtype(dataframe)
            target_exists = self.validate_target_column_exists(dataframe)
            target_status, target_label_issues = self.validate_target_inputs(dataframe)
            missing_status, missing_values = self.check_missing_values(dataframe)
            whitespace_status, whitespace_issues = self.validate_whitespace_text(dataframe)
            duplicate_status, duplicate_count = self.check_duplicates(dataframe)
            _, class_distribution = self.validate_class_distribution(dataframe)

            logger.info("All validation checks completed. Generating validation report.")

            overall_status = all([
                col_status,
                unexpected_status,
                dtype_status,
                target_exists,
                target_status,
                missing_status,
                whitespace_status,
                duplicate_status,
            ])

            self.write_validation_report(
                dataframe=dataframe,
                overall_status=overall_status,
                missing_columns=missing_columns,
                unexpected_columns=unexpected_cols,
                dtype_mismatches=dtype_mismatches,
                target_exists=target_exists,
                target_label_issues=target_label_issues,
                missing_values=missing_values,
                whitespace_issues=whitespace_issues,
                duplicate_count=duplicate_count,
                class_distribution=class_distribution,
            )

            logger.info(
                    f"Data validation completed. Overall status: "
                    f"{'PASSED' if overall_status else 'FAILED'}."
                )

            return overall_status

        except Exception as e:
            raise CustomException(e, sys)
        


        