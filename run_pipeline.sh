#!/bin/bash

echo "Creating load data..."
python create_load_csv.py

echo "Creating solar data..."
python create_solar_csv.py

echo "Training model..."
python train.py

echo "Evaluating model..."
python evaluate.py

echo "Running trade-off experiment..."
python tradeoff_experiment.py

echo "Pipeline completed!"