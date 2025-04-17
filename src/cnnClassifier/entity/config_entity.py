from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path
    



@dataclass(frozen=True)
class TrainingConfig:
    root_dir: Path
    trained_model_path: Path
    training_data: Path
    params_epochs: int
    params_batch_size: int
    params_is_augmentation: bool
    params_image_size: list
    # opciones de base model
    params_weights: str
    params_include_top: bool
    params_classes: int
    # tuner
    tuner_max_trials: int
    tuner_executions_per_trial: int
    hp_learning_rate: dict
    hp_freeze_till: dict
    hp_dense_units: dict



@dataclass(frozen=True)
class EvaluationConfig:
    path_of_model: Path
    training_data: Path
    all_params: dict
    mlflow_uri: str
    params_image_size: list
    params_batch_size: int