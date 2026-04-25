import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .config import OUTPUT_DIR, EDA_DIR, PLOTS_DIR, ARTIFACT_DIR


def ensure_dirs() -> None:
    for p in [OUTPUT_DIR, EDA_DIR, PLOTS_DIR, ARTIFACT_DIR]:
        os.makedirs(p, exist_ok=True)


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)


def savefig(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()


def save_json(obj: Any, path: Path) -> None:
    def default(o):
        if isinstance(o, (np.integer, np.floating)):
            return o.item()
        if isinstance(o, (pd.Timestamp,)):
            return str(o)
        return str(o)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, default=default, indent=2)


def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)