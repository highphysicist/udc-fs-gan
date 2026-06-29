#!/usr/bin/env bash
set -euo pipefail
python -m evaluation.eval_udc_fsgan --config configs/eval.yaml --checkpoint weights/udc_fsgan.pt
