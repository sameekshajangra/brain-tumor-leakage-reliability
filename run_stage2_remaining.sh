#!/bin/bash
set -e

echo "Starting full Stage 2 execution..."

echo "Training Condition A..."
python3 src/10_train_efficientnet.py --condition A

echo "Training Condition B..."
python3 src/10_train_efficientnet.py --condition B

echo "Training Condition C..."
python3 src/10_train_efficientnet.py --condition C

echo "Generating predictions..."
python3 src/11_generate_predictions.py

echo "Computing metrics..."
python3 src/12_compute_metrics.py

echo "Computing calibration..."
python3 src/13_calibration.py

echo "Generating figures and report..."
python3 src/14_generate_pilot_figures.py

echo "Stage 2 execution complete!"
