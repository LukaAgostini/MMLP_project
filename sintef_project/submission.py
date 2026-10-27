"""Helpers for creating Kaggle submissions."""

from __future__ import annotations

from os import PathLike
from pathlib import Path
import re

import numpy as np
import pandas as pd


_DEFAULT_SAMPLE_SUBMISSION = Path(__file__).with_name("sample_submission.csv")
_DEFAULT_PREDICTION_MAPPING = Path(__file__).with_name("prediction_mapping.csv")
_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def create_submission(
    predictions: pd.DataFrame | np.ndarray,
    identifier: str = "submission",
    output_dir: str | PathLike[str] = "output",
    sample_submission_path: str | PathLike[str] = _DEFAULT_SAMPLE_SUBMISSION,
    prediction_mapping_path: str | PathLike[str] = _DEFAULT_PREDICTION_MAPPING,
) -> Path:
    """Create and save a Kaggle submission CSV.

    ``predictions`` may either be a DataFrame with the prediction column names
    or a two-dimensional array whose columns follow the sample submission.
    When the DataFrame contains ``Run No``, its identifiers are validated and
    used; otherwise rows are matched to ``prediction_mapping.csv`` by position.

    Args:
        predictions: Predicted probabilities, one row per test run.
        identifier: File stem used for ``<identifier>.csv``.
        output_dir: Directory in which the submission file is written.
        sample_submission_path: Kaggle sample submission template.
        prediction_mapping_path: Mapping of test rows to ``Run No`` values.

    Returns:
        The path of the written submission file.

    Raises:
        ValueError: If the predictions do not match the Kaggle schema.
        FileNotFoundError: If a referenced template or mapping is missing.
    """
    if not _IDENTIFIER_PATTERN.fullmatch(identifier):
        raise ValueError(
            "identifier must contain only letters, numbers, '.', '_' or '-'; "
            "it must not contain a path separator."
        )

    sample_path = Path(sample_submission_path)
    mapping_path = Path(prediction_mapping_path)
    sample = pd.read_csv(sample_path)
    mapping = pd.read_csv(mapping_path)

    _validate_reference_files(sample, mapping)
    label_columns = [column for column in sample.columns if column != "Run No"]

    if isinstance(predictions, pd.DataFrame):
        prediction_frame = predictions.copy()
        if "Run No" in prediction_frame.columns:
            _validate_run_numbers(prediction_frame["Run No"], mapping["Run No"])
            prediction_frame = prediction_frame.set_index("Run No")
            prediction_frame = prediction_frame.reindex(mapping["Run No"])
        else:
            prediction_frame = prediction_frame.reset_index(drop=True)
    elif isinstance(predictions, np.ndarray):
        if predictions.ndim != 2:
            raise ValueError("predictions must be a two-dimensional array.")
        prediction_frame = pd.DataFrame(predictions, columns=label_columns)
    else:
        raise TypeError("predictions must be a pandas DataFrame or numpy array.")

    if len(prediction_frame) != len(mapping):
        raise ValueError(
            f"Expected {len(mapping)} prediction rows, got {len(prediction_frame)}."
        )

    missing_columns = [column for column in label_columns if column not in prediction_frame]
    extra_columns = [
        column for column in prediction_frame.columns if column not in label_columns
    ]
    if missing_columns or extra_columns:
        raise ValueError(
            f"Prediction columns do not match the sample submission. "
            f"Missing: {missing_columns[:5]}; extra: {extra_columns[:5]}."
        )

    prediction_frame = prediction_frame[label_columns]
    if not prediction_frame.apply(pd.api.types.is_numeric_dtype).all():
        raise ValueError("All prediction columns must contain numeric values.")
    if not np.isfinite(prediction_frame.to_numpy(dtype=float)).all():
        raise ValueError("Predictions must contain only finite values.")

    submission = sample.copy()
    submission[label_columns] = prediction_frame.to_numpy()
    submission["Run No"] = mapping["Run No"].to_numpy()

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    submission_path = output_path / f"{identifier}.csv"
    submission.to_csv(submission_path, index=False)
    return submission_path


def _validate_reference_files(sample: pd.DataFrame, mapping: pd.DataFrame) -> None:
    if "Run No" not in sample.columns:
        raise ValueError("The sample submission must contain a 'Run No' column.")
    if list(mapping.columns) != ["Run No", "starttime"]:
        raise ValueError(
            "The prediction mapping must contain exactly 'Run No' and 'starttime'."
        )
    if len(sample) != len(mapping):
        raise ValueError(
            "The sample submission and prediction mapping must have the same number "
            "of rows."
        )
    if sample["Run No"].duplicated().any() or mapping["Run No"].duplicated().any():
        raise ValueError("Run No values must be unique.")
    if not sample["Run No"].equals(mapping["Run No"]):
        raise ValueError(
            "The Run No order in the sample submission and prediction mapping differs."
        )


def _validate_run_numbers(
    prediction_ids: pd.Series, expected_ids: pd.Series
) -> None:
    if prediction_ids.isna().any() or prediction_ids.duplicated().any():
        raise ValueError("Prediction Run No values must be present and unique.")
    if set(prediction_ids) != set(expected_ids):
        raise ValueError("Prediction Run No values do not match the test mapping.")
