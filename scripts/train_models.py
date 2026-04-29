"""Train AC and HP detector models from the unified Dataport training dataset.

Reads a pre-processed parquet that combines smart meter load features with
Dataport ground-truth appliance labels and writes trained scikit-learn models
to the ``models/`` directory. Existing artifacts are overwritten.

Prerequisites:
    - Training data at the path set by ``data.training_data_path`` in the config
      (default: ``data/processed/training/all_sources_load_with_weather.parquet``).
      Required columns include ``customer_id``, ``label``, and all feature columns
      expected by ACDetectorTrainer / HPDetectorTrainer (see their FEATURE_COLS).
    - scikit-learn 1.3.x — model artifacts are not forward-compatible across minor
      versions (pinned in requirements.txt).

Output:
    models/ac_detector_v1.joblib   — Random-forest AC detector (sklearn Pipeline)
    models/hp_detector_v1.joblib   — Random-forest HP detector (sklearn Pipeline)

Usage:
    python scripts/train_models.py --config config/re_production.yaml
    python scripts/train_models.py --models ac_detector
    python scripts/train_models.py --training-data path/to/training.parquet
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from re_nilm.pipeline.orchestrator import load_config
from re_nilm.training.trainers.ac_detector_trainer import ACDetectorTrainer, FEATURE_COLS as AC_FEATURES
from re_nilm.training.trainers.hp_detector_trainer import HPDetectorTrainer, FEATURE_COLS as HP_FEATURES


def _parse_args():
    parser = argparse.ArgumentParser(description="Train RE-NILM appliance detector models")
    parser.add_argument("--config", default="config/re_production.yaml")
    parser.add_argument(
        "--models",
        default="ac_detector,hp_detector",
        help="Comma-separated list of models to train",
    )
    parser.add_argument(
        "--training-data",
        default=None,
        help="Path to all_sources_load_with_weather.parquet (overrides config)",
    )
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def _load_training_data(path: Path) -> "pd.DataFrame":
    """Load the training parquet and raise a clear error if the file is missing."""
    import pandas as pd
    if not path.exists():
        raise FileNotFoundError(f"Training data not found: {path}")
    return pd.read_parquet(path)


def _train_ac_detector(training_data_path: Path, output_path: Path, cfg: dict) -> None:
    """Fit, evaluate, and save the AC detector model."""
    import pandas as pd

    logger = logging.getLogger("train_models.ac")
    logger.info("Loading training data from %s", training_data_path)
    df = _load_training_data(training_data_path)
    logger.info("Loaded %d rows, columns: %s", len(df), df.columns.tolist())

    trainer = ACDetectorTrainer(
        test_size=0.2,
        rf_params=None,
    )
    trainer.fit(df)

    metrics = trainer.evaluate()
    logger.info("AC detector evaluation:\n%s", metrics.get("report", ""))
    logger.info("F1=%.3f on %d test customers", metrics["f1"], metrics["n_test"])

    trainer.save(output_path)
    logger.info("AC detector saved → %s", output_path)


def _train_hp_detector(training_data_path: Path, output_path: Path, cfg: dict) -> None:
    """Fit, evaluate, and save the heat-pump detector model."""
    logger = logging.getLogger("train_models.hp")
    logger.info("Loading training data from %s", training_data_path)
    df = _load_training_data(training_data_path)
    logger.info("Loaded %d rows, columns: %s", len(df), df.columns.tolist())

    trainer = HPDetectorTrainer(test_size=0.2)
    trainer.fit(df)

    metrics = trainer.evaluate()
    logger.info("HP detector evaluation:\n%s", metrics.get("report", ""))
    logger.info("F1=%.3f on %d test customers", metrics["f1"], metrics["n_test"])

    trainer.save(output_path)
    logger.info("HP detector saved → %s", output_path)


def main():
    args = _parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    cfg = load_config(args.config)
    models_to_train = [m.strip() for m in args.models.split(",")]

    training_data_path = Path(
        args.training_data
        or cfg.get("data", {}).get("training_data_path", "data/processed/training/all_sources_load_with_weather.parquet")
    )
    models_dir = Path(cfg.get("output", {}).get("models_dir", "models"))
    models_dir.mkdir(parents=True, exist_ok=True)

    if "ac_detector" in models_to_train:
        _train_ac_detector(
            training_data_path,
            models_dir / "ac_detector_v1.joblib",
            cfg,
        )

    if "hp_detector" in models_to_train:
        _train_hp_detector(
            training_data_path,
            models_dir / "hp_detector_v1.joblib",
            cfg,
        )

    logging.getLogger(__name__).info("Training complete.")


if __name__ == "__main__":
    main()
