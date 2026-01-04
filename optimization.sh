#!/bin/bash

SEED=42
N_TRIALS=20
EPOCHS=50
PATIENCE=15
KFOLDS=5
TEST_SIZE=0.2

for DATASET in DEAM PMEmo MERGE; do
  for MODE in VA Russell4Q; do
    for HEAD in BiGRU CNNLSTM; do
      echo "=== Running: $DATASET + $MODE + $HEAD ==="

      CMD="python -m mer.modeling.optimize \
        --dataset-name $DATASET \
        --prediction-mode $MODE \
        --head $HEAD \
        --n-trials $N_TRIALS \
        --seed $SEED \
        --test-size $TEST_SIZE \
        --epochs $EPOCHS \
        --patience $PATIENCE \
        --kfolds $KFOLDS"

      if [ "$DATASET" = "MERGE" ]; then
        CMD="$CMD --merge-split 70_15_15"
      else
        CMD="$CMD --labels-scale norm"
      fi

      if [ "$MODE" = "VA" ]; then
        CMD="$CMD --loss-type ccc"
      fi

      eval $CMD
    done
  done
done