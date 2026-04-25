from __future__ import annotations

from pathlib import Path
from typing import Literal

import dspy
import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


class TableFactCheckSignature(dspy.Signature):
    """
    Verify the given claim against the provided table data.
    """

    claim: str = dspy.InputField(desc="Claim to be verified")
    table: str = dspy.InputField(desc="Table data")
    caption: str = dspy.InputField(desc="Table caption")

    answer: Literal["supports", "refutes", "not enough info"] = dspy.OutputField(
        desc="Whether the claim is supported, refuted or cannot be verified by the table data."
    )


class TableFactCheckPoTSignature(dspy.Signature):
    """
    Verify the given claim against the provided table data. Implement a python program to help with data processing and analysis. Wrap up your python code with ```python and ``` and make sure you don't use any external libraries.
    """

    claim: str = dspy.InputField(desc="Claim to be verified")
    table: str = dspy.InputField(desc="Table data")
    caption: str = dspy.InputField(desc="Table caption")

    answer: Literal["supports", "refutes", "not enough info"] = dspy.OutputField(
        desc="Whether the claim is supported, refuted or cannot be verified by the table data."
    )


def load_config(config_path: str | None = None):
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()

    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    _normalise_project_paths(config)
    config["_config_path"] = str(path)
    config["_project_root"] = str(PROJECT_ROOT)
    return config


def _normalise_project_paths(config):
    for key, value in config.get("paths", {}).items():
        if key.endswith("_dir") and isinstance(value, str):
            config["paths"][key] = _resolve_from_project_root(value)

    for dataset_config in config.get("datasets", {}).values():
        for key, value in dataset_config.items():
            if key.endswith("_path") and isinstance(value, str):
                dataset_config[key] = _resolve_from_project_root(value)


def _resolve_from_project_root(path_str: str) -> str:
    path = Path(path_str).expanduser()
    if path.is_absolute():
        return path.as_posix()
    return (PROJECT_ROOT / path).resolve().as_posix()
