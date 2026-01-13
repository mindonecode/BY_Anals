#!/bin/bash

BASE_DIR=$(cd "$(dirname "$0")" && pwd)

source "$BASE_DIR/venv/bin/activate"
python "$BASE_DIR/main.py"