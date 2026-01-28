#!/usr/bin/env bash

set -e  # fail fast on errors

# ===== Configuration =====
NOTEBOOK="Assignment2_HourlyPowerGenerationofEurope.ipynb"
BASENAME="${NOTEBOOK%.ipynb}"
OUTPUT_DIR="./export"

LATEX_FILE="${OUTPUT_DIR}/${BASENAME}.tex"
LATEX_NOPYP_FILE="${OUTPUT_DIR}/${BASENAME}_no_python.tex"

# ===== Ensure output directory exists =====
mkdir -p "$OUTPUT_DIR"

# ===== Export formats =====
jupyter nbconvert --to markdown "$NOTEBOOK" --output-dir="$OUTPUT_DIR"
jupyter nbconvert --to python "$NOTEBOOK" --output-dir="$OUTPUT_DIR"
jupyter nbconvert --to latex "$NOTEBOOK" --output-dir="$OUTPUT_DIR"
jupyter nbconvert --to html   --no-input "$NOTEBOOK"  --output-dir="$OUTPUT_DIR"
# jupyter nbconvert --to html  --clear-output --no-input "$NOTEBOOK"  --output-dir="$OUTPUT_DIR"
# jupyter nbconvert --to html --template external_images --FilesWriter.build_directory=notebook_files --output notebook.html "$NOTEBOOK"
# jupyter nbconvert --to html --template external_images --FilesWriter.build_directory="$OUTPUT_DIR" --output "$BASENAME.html" "$NOTEBOOK" --template-path="./templates"


# ===== Remove Python code blocks (Highlighting) BEFORE cleaning =====
sed '/\\begin{Verbatim}/,/\\end{Verbatim}/d' "$LATEX_FILE" > "$LATEX_NOPYP_FILE"

# ===== Clean notebook AFTER exports =====
nbstripout "$NOTEBOOK"

echo "LaTeX without Python code written to:"
echo "  $LATEX_NOPYP_FILE"

echo "Actual tree:"
tree -h .
