#!/bin/bash
set -e
echo "Running data processing..."
python src/01_data_processing.py
echo "Running 01-data-exploration.ipynb ..."
mkdir -p /app/data/results
jupyter nbconvert --to notebook --execute notebook/01-data-exploration.ipynb --output /app/data/results/01-data-exploration-executed.ipynb
echo "Running 02-label-analysis.ipynb ..."
jupyter nbconvert --to notebook --execute notebook/02-label-analysis.ipynb --output /app/data/results/02-label-analysis-executed.ipynb
echo "Running model training..."
python src/02_training.py
echo "Running evaluation..."
python src/03_evaluation.py
echo "Running inference..."
python src/04_inference.py
echo "Pipeline finished successfully."