from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd


def metrics_to_markdown(results: Dict[str, Dict[str, float]]) -> str:
    df = pd.DataFrame(results).T
    return df.to_markdown(floatfmt='.4f')


def save_metrics_report(results: Dict[str, Dict[str, float]], out_path: str):
    text = metrics_to_markdown(results)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(text, encoding='utf-8')
