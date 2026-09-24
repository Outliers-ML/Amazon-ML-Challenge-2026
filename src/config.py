"""Configuration management module for the ML pipeline.

Provides structured dataclasses and YAML loading utilities to enforce typed,
reproducible experiment configurations across models, datasets, and pipelines.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml


@dataclass
class PathConfig:
    """Paths used throughout the pipeline.

    Attributes:
        data_dir: Base directory for datasets.
        raw_data_dir: Path to raw dataset files.
        processed_data_dir: Path to preprocessed data artifacts.
        checkpoints_dir: Directory where trained model checkpoints are stored.
        submissions_dir: Directory where test predictions and submission files are saved.
        logs_dir: Directory where execution logs are written.
    """
    data_dir: Path = Path("data")
    raw_data_dir: Path = Path("data/raw")
    processed_data_dir: Path = Path("data/processed")
    checkpoints_dir: Path = Path("checkpoints")
    submissions_dir: Path = Path("submissions")
    logs_dir: Path = Path("logs")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PathConfig":
        """Instantiate PathConfig from a dictionary with path string conversion."""
        return cls(
            data_dir=Path(data.get("data_dir", "data")),
            raw_data_dir=Path(data.get("raw_data_dir", "data/raw")),
            processed_data_dir=Path(data.get("processed_data_dir", "data/processed")),
            checkpoints_dir=Path(data.get("checkpoints_dir", "checkpoints")),
            submissions_dir=Path(data.get("submissions_dir", "submissions")),
            logs_dir=Path(data.get("logs_dir", "logs")),
        )


@dataclass
class DataConfig:
    """Dataset configuration parameters.

    Attributes:
        train_file: Filename of the training dataset.
        test_file: Filename of the test dataset.
        target_col: Name of target prediction column(s).
        id_col: Primary key/identifier column.
        text_cols: List of textual feature columns.
        numerical_cols: List of numeric feature columns.
        categorical_cols: List of categorical feature columns.
        image_cols: List of image path/URL columns if multimodal.
    """
    train_file: str = "train.csv"
    test_file: str = "test.csv"
    target_col: str = "target"
    id_col: str = "id"
    text_cols: List[str] = field(default_factory=list)
    numerical_cols: List[str] = field(default_factory=list)
    categorical_cols: List[str] = field(default_factory=list)
    image_cols: List[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DataConfig":
        """Instantiate DataConfig from dictionary."""
        return cls(
            train_file=data.get("train_file", "train.csv"),
            test_file=data.get("test_file", "test.csv"),
            target_col=data.get("target_col", "target"),
            id_col=data.get("id_col", "id"),
            text_cols=data.get("text_cols", []),
            numerical_cols=data.get("numerical_cols", []),
            categorical_cols=data.get("categorical_cols", []),
            image_cols=data.get("image_cols", []),
        )


@dataclass
class ModelConfig:
    """Model architecture and hyperparameter settings.

    Attributes:
        model_type: Name of the model architecture (e.g. 'lightgbm', 'xgboost', 'catboost', 'nn').
        params: Dictionary of hyperparameters passed to the model constructor.
        early_stopping_rounds: Rounds of patience before early stopping triggers.
        num_boost_round: Maximum number of boosting rounds / epochs.
    """
    model_type: str = "lightgbm"
    params: Dict[str, Any] = field(default_factory=dict)
    early_stopping_rounds: int = 50
    num_boost_round: int = 1500

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelConfig":
        """Instantiate ModelConfig from dictionary."""
        return cls(
            model_type=data.get("model_type", "lightgbm"),
            params=data.get("params", {}),
            early_stopping_rounds=data.get("early_stopping_rounds", 50),
            num_boost_round=data.get("num_boost_round", 1500),
        )


@dataclass
class TrainConfig:
    """Training, cross-validation and optimization controls.

    Attributes:
        seed: Random seed for universal reproducibility.
        n_splits: Number of cross-validation folds.
        stratified: Whether to use StratifiedKFold for classification.
        group_col: Column name to group splits by (for GroupKFold), if any.
        metric: Primary evaluation metric (e.g. 'f1_macro', 'accuracy', 'rmse', 'mape').
        device: Computing device ('cuda' or 'cpu').
        num_workers: Data loader worker processes.
        batch_size: Batch size for neural network models.
    """
    seed: int = 42
    n_splits: int = 5
    stratified: bool = True
    group_col: Optional[str] = None
    metric: str = "f1_macro"
    device: str = "cuda"
    num_workers: int = 4
    batch_size: int = 64

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrainConfig":
        """Instantiate TrainConfig from dictionary."""
        return cls(
            seed=data.get("seed", 42),
            n_splits=data.get("n_splits", 5),
            stratified=data.get("stratified", True),
            group_col=data.get("group_col", None),
            metric=data.get("metric", "f1_macro"),
            device=data.get("device", "cuda"),
            num_workers=data.get("num_workers", 4),
            batch_size=data.get("batch_size", 64),
        )


@dataclass
class CompetitionConfig:
    """Master experiment configuration encapsulating all sub-configurations.

    Attributes:
        experiment_name: Descriptive name of the current experiment run.
        paths: File and directory paths configuration.
        data: Dataset attributes and column specifications.
        model: Model architecture and hyperparameter settings.
        train: Training regime and validation settings.
    """
    experiment_name: str
    paths: PathConfig = field(default_factory=PathConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    @classmethod
    def from_yaml(cls, yaml_path: str | Path) -> "CompetitionConfig":
        """Load configuration from a YAML file.

        Args:
            yaml_path: Path to the YAML configuration file.

        Returns:
            CompetitionConfig instance populated with settings from YAML.

        Raises:
            FileNotFoundError: If the YAML file does not exist.
        """
        path = Path(yaml_path)
        if not path.is_file():
            raise FileNotFoundError(f"Configuration file not found at: {path.resolve()}")

        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        return cls(
            experiment_name=data.get("experiment_name", "baseline"),
            paths=PathConfig.from_dict(data.get("paths", {})),
            data=DataConfig.from_dict(data.get("data", {})),
            model=ModelConfig.from_dict(data.get("model", {})),
            train=TrainConfig.from_dict(data.get("train", {})),
        )
