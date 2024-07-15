#!/usr/bin/env bash

set -x

OUTPUT_DIR=output/dosa
ARGS=${@:1}

python -u main.py \
    --output_dir ${OUTPUT_DIR} \
    ${ARGS}
