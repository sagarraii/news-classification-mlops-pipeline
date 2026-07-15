from dataclasses import dataclass

@dataclass(frozen=True)
class DataValidationArtifact:
    validation_status: bool
    status_file_path: str