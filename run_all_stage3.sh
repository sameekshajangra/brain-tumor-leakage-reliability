#!/bin/bash
echo "Starting 15-run Stage 3 Experiment"
for fold in 1 2 3 4 5; do
    for cond in A B C; do
        python3 src/19_stage3_train.py --fold $fold --condition $cond
    done
done
echo "All 15 runs completed."
