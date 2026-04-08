#!/bin/bash

echo "Step 1: Generating load data..."
python create_load_csv.py

echo "Step 2: Generating solar data..."
python create_solar_csv.py

echo "Step 3: Training models..."
python train.py

echo "Step 4: Averaging results..."
python evaluate.py

echo "Step 5: Running trade-off experiment..."
python tradeoff_experiment.py

echo "Pipeline completed successfully!"