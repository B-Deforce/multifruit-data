#!/bin/bash
# run from project root
set -e

data_config_name="bdb/bdb"

python main.py \
    --config_name "$data_config_name" \
    --dataloader bdb \