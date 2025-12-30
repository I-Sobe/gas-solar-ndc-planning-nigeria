"""
I/O Utilities

Low-level helpers for loading configuration files and
persisting numerical model outputs.

This module does not perform plotting or reporting.
"""

import os
import yaml
import numpy as np


def load_yaml(filepath):
    """
    Load a YAML configuration file.

    Parameters
    ----------
    filepath : str
        Path to YAML file

    Returns
    -------
    dict
        Parsed YAML contents
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"YAML file not found: {filepath}")

    with open(filepath, "r") as f:
        return yaml.safe_load(f)


def save_results(output, filepath):
    """
    Save numerical model results to disk.

    Parameters
    ----------
    output : dict or array-like
        Model results to persist
    filepath : str
        Output file path ('.npz' or '.npy')

    Returns
    -------
    None
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    if isinstance(output, dict):
        np.savez(filepath, **output)
    else:
        np.save(filepath, output)
