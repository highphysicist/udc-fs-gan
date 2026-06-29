from __future__ import annotations

from pathlib import Path
import json
import yaml


def load_yaml(path: str):
    return yaml.safe_load(Path(path).read_text(encoding='utf-8'))


def save_yaml(data, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(yaml.safe_dump(data, sort_keys=False), encoding='utf-8')


def load_json(path: str):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save_json(data, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, indent=2), encoding='utf-8')
