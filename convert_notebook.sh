#!/usr/bin/env bash

# ===== Copy Jupyter File =====
cp ~/OneDrive/Courses/STM_WS2025_DA_Data\ Analysis/Assignment_2/Assignment2_HourlyPowerGenerationofEurope.ipynb ~/workspace/STM_WS2025_DA_Assignment2/Assignment2_HourlyPowerGenerationofEurope.ipynb

# ===== Configuration =====
NOTEBOOK="Assignment2_HourlyPowerGenerationofEurope.ipynb"
OUTPUT_DIR="./export"

# ===== Ensure output directory exists =====
mkdir -p "$OUTPUT_DIR"

# ===== Commands =====

jupyter nbconvert --to markdown "$NOTEBOOK" --output-dir="$OUTPUT_DIR"
jupyter nbconvert --to html "$NOTEBOOK" --output-dir="$OUTPUT_DIR"
jupyter nbconvert --to python "$NOTEBOOK" --output-dir="$OUTPUT_DIR"
jupyter nbconvert --to latex "$NOTEBOOK" --output-dir="$OUTPUT_DIR"

nbstripout "$NOTEBOOK"
#To see all available configurables, use `--help-all`.
