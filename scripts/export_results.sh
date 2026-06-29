#!/usr/bin/env bash
set -euo pipefail
python -m evaluation.eval_comparisons --config configs/compare_models.yaml
