#!/usr/bin/env bash
set -euo pipefail
python -m training.train_udc_fsgan --config configs/celeba_hq_4x.yaml
