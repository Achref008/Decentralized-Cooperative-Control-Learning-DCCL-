# src/data_loader.py
"""
Dataset loading and basic cleaning utilities.

This module is responsible for:
- Loading tabular data from an Excel file.
- Performing minimal, reproducible cleaning.
- Partitioning the dataset into balanced chunks for distributed nodes.

It does NOT perform feature engineering or scaling.
Those steps are handled in `preprocessing.py`.
"""

import logging
from typing import List

import numpy as np
import pandas as pd

# Use project-level logging configuration (do not reconfigure here)
logger = logging.getLogger(__name__)


# ================================================================
# CORE DATA LOADING
# ================================================================

def load_data(file_path: str) -> pd.DataFrame:
    """
    Load an Excel dataset and perform basic validation.

    Args:
        file_path: Path to the `.xlsx` dataset.

    Returns:
        A cleaned pandas DataFrame.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is empty or unreadable.
    """
    try:
        df = pd.read_excel(file_path)
    except Exception as e:
        raise ValueError(f"Failed to read Excel file: {e}") from e

    if df is None or df.empty:
        raise ValueError(f"Loaded dataset is empty: {file_path}")

    # Normalize column names (no side effects outside this function)
    df = df.copy()
    df.columns = df.columns.str.strip()

    logger.info(
        "Loaded dataset from %s with shape %s and %d columns.",
        file_path,
        df.shape,
        len(df.columns),
    )
    logger.debug("Columns: %s", list(df.columns))

    return df


# ================================================================
# MISSING VALUE HANDLING
# ================================================================

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute missing values in a conservative, reproducible way.

    - Numeric columns → mean imputation.
    - Categorical columns → most frequent value (mode).

    The function returns a new DataFrame (no in-place modification).

    Args:
        df: Input DataFrame.

    Returns:
        Cleaned DataFrame with no missing values.
    """
    df = df.copy()

    # Numeric columns
    num_cols = df.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        if df[col].isna().any():
            mean_value = df[col].mean()
            df[col] = df[col].fillna(mean_value)
            logger.info(
                "Imputed missing values in numeric column '%s' using mean=%.6f.",
                col,
                mean_value,
            )

    # Categorical columns
    cat_cols = df.select_dtypes(include=["object"]).columns
    for col in cat_cols:
        if df[col].isna().any():
            mode_value = df[col].mode().iloc[0]
            df[col] = df[col].fillna(mode_value)
            logger.info(
                "Imputed missing values in categorical column '%s' using mode='%s'.",
                col,
                mode_value,
            )

    return df


# ================================================================
# DATA PARTITIONING FOR NODES
# ================================================================

def split_data_balanced(df: pd.DataFrame, num_nodes: int) -> List[pd.DataFrame]:
    """
    Partition the dataset into approximately equal-sized chunks.

    This function preserves row order within each chunk but
    does not shuffle the data.

    Args:
        df: Input DataFrame.
        num_nodes: Number of partitions (distributed nodes).

    Returns:
        List of DataFrame chunks, one per node.
    """
    if num_nodes <= 0:
        raise ValueError("num_nodes must be a positive integer.")

    chunks = np.array_split(df, num_nodes)

    for i, chunk in enumerate(chunks):
        logger.info(
            "Node %d: assigned %d samples (%d columns).",
            i,
            chunk.shape[0],
            chunk.shape[1],
        )

    return chunks


# ================================================================
# END-TO-END PIPELINE
# ================================================================

def preprocess_and_split_data(file_path: str, num_nodes: int) -> List[pd.DataFrame]:
    """
    Load, clean, and partition the dataset for distributed training.

    This function performs only:
    1) Loading,
    2) Missing-value handling,
    3) Balanced partitioning.

    Feature engineering, scaling, and train/test splits are handled
    in `preprocessing.py`.

    Args:
        file_path: Path to the Excel dataset.
        num_nodes: Number of distributed nodes.

    Returns:
        List of DataFrames, one per node.
    """
    df = load_data(file_path)
    df = handle_missing_values(df)
    node_chunks = split_data_balanced(df, num_nodes)
    return node_chunks
