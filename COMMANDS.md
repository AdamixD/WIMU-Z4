# MER Commands Reference

## Embeddings Extraction

```bash
# DEAM
python -m mer.modeling.embeddings \
  --dataset-name DEAM

# PMEmo
python -m mer.modeling.embeddings \
  --dataset-name PMEmo

# MERGE
python -m mer.modeling.embeddings \
  --dataset-name MERGE

# With augments
python -m mer.modeling.embeddings \
  --dataset-name PMEmo \ 
  --augment shift 
```

## Training

### DEAM VA
```bash
python -m mer.modeling.train \
  --dataset-name DEAM \
  --prediction-mode VA \
  --kfolds 5 \
  --epochs 100 \
  --batch-size 8 \
  --lr 0.001 \
  --hidden-dim 128 \
  --dropout 0.2 \
  --patience 15 \
  --loss-type ccc \
  --labels-scale norm \
  --test-size 0.1 \
  --seed 42
```

### DEAM Russell4Q
```bash
python -m mer.modeling.train \
  --dataset-name DEAM \
  --prediction-mode Russell4Q \
  --kfolds 5 \
  --epochs 100 \
  --batch-size 8 \
  --lr 0.001 \
  --hidden-dim 128 \
  --dropout 0.2 \
  --patience 15 \
  --labels-scale norm \
  --test-size 0.1 \
  --seed 42
```

### PMEmo VA
```bash
python -m mer.modeling.train \
  --dataset-name PMEmo \
  --prediction-mode VA \
  --kfolds 5 \
  --epochs 100 \
  --batch-size 8 \
  --lr 0.001 \
  --hidden-dim 128 \
  --dropout 0.2 \
  --patience 15 \
  --loss-type ccc \
  --labels-scale norm \
  --test-size 0.1 \
  --seed 42
```

### PMEmo Russell4Q
```bash
python -m mer.modeling.train \
  --dataset-name PMEmo \
  --prediction-mode Russell4Q \
  --kfolds 5 \
  --epochs 100 \
  --batch-size 8 \
  --lr 0.001 \
  --hidden-dim 128 \
  --dropout 0.2 \
  --patience 15 \
  --labels-scale norm \
  --test-size 0.1 \
  --seed 42
```

### MERGE VA
```bash
python -m mer.modeling.train \
  --dataset-name MERGE \
  --prediction-mode VA \
  --merge-split 70_15_15 \
  --epochs 100 \
  --batch-size 6 \
  --lr 0.001 \
  --hidden-dim 128 \
  --dropout 0.2 \
  --patience 15 \
  --loss-type ccc
```

### MERGE Russell4Q
```bash
python -m mer.modeling.train \
  --dataset-name MERGE \
  --prediction-mode Russell4Q \
  --merge-split 70_15_15 \
  --epochs 100 \
  --batch-size 6 \
  --lr 0.001 \
  --hidden-dim 128 \
  --dropout 0.2 \
  --patience 15
```

### Augments
Possible augments: shift, gain, reverb, lowpass, highpass, bandpass, pitch_shift
```bash
python -m mer.modeling.train \
  --dataset-name PMEmo \
  --prediction-mode VA \
  --kfolds 5 \
  --epochs 100 \
  --batch-size 8 \
  --lr 0.001 \
  --hidden-dim 128 \
  --dropout 0.2 \
  --patience 15 \
  --loss-type ccc \
  --labels-scale norm \
  --test-size 0.1 \
  --seed 42 \
  --augments shift \
  --augments gain \
  --augment-size 0.3

```

## Evaluation

### DEAM VA
```bash
python -m mer.modeling.evaluate \
  reports/training_DEAM_VA_<timestamp>/model.pth \
  --dataset-name DEAM \
  --split test \
  --test-size 0.1 \
  --seed 42 \
  --labels-scale norm
```

### DEAM VA with Russell4Q mapping
```bash
python -m mer.modeling.evaluate \
  reports/training_DEAM_VA_<timestamp>/model.pth \
  --dataset-name DEAM \
  --split test \
  --test-size 0.1 \
  --seed 42 \
  --labels-scale norm \
  --map-to-russell4q
```

### DEAM Russell4Q
```bash
python -m mer.modeling.evaluate \
  reports/training_DEAM_Russell4Q_<timestamp>/model.pth \
  --dataset-name DEAM \
  --split test \
  --test-size 0.1 \
  --seed 42 \
  --labels-scale norm
```

### PMEmo VA
```bash
python -m mer.modeling.evaluate \
  reports/training_PMEmo_VA_<timestamp>/model.pth \
  --dataset-name PMEmo \
  --split test \
  --test-size 0.1 \
  --seed 42 \
  --labels-scale norm
```

### PMEmo Russell4Q
```bash
python -m mer.modeling.evaluate \
  reports/training_PMEmo_Russell4Q_<timestamp>/model.pth \
  --dataset-name PMEmo \
  --split test \
  --test-size 0.1 \
  --seed 42 \
  --labels-scale norm
```

### MERGE VA
```bash
python -m mer.modeling.evaluate \
  reports/training_MERGE_VA_<timestamp>/model.pth \
  --dataset-name MERGE \
  --split test \
  --merge-split 70_15_15
```

### MERGE Russell4Q
```bash
python -m mer.modeling.evaluate \
  reports/training_MERGE_Russell4Q_<timestamp>/model.pth \
  --dataset-name MERGE \
  --split test \
  --merge-split 70_15_15
```

## Prediction

**Model type (VA/Russell4Q) is automatically detected from output layer size.**

**Available flags:**
- `--return-average`: Compute and save track-level statistics (JSON file)
- `--verbose`: Display detailed frame-by-frame statistics
- `--map-to-russell4q`: Map VA predictions to Russell 4Q quadrants (VA models only)

### VA Prediction (frame-by-frame)
```bash
python -m mer.modeling.predict \
  --model-path reports/training_<dataset>_VA_<timestamp>/model.pth \
  --audio-path audio.mp3 \
  --output-path predictions.csv
```

### VA Prediction (track average)
```bash
python -m mer.modeling.predict \
  --model-path reports/training_<dataset>_VA_<timestamp>/model.pth \
  --audio-path audio.mp3 \
  --output-path predictions_avg.csv \
  --return-average
```

### VA Prediction with Russell4Q mapping
```bash
python -m mer.modeling.predict \
  --model-path reports/training_<dataset>_VA_<timestamp>/model.pth \
  --audio-path audio.mp3 \
  --output-path predictions_r4q.csv \
  --map-to-russell4q
```

### Russell4Q Prediction (frame-by-frame)
```bash
python -m mer.modeling.predict \
  --model-path reports/training_<dataset>_Russell4Q_<timestamp>/model.pth \
  --audio-path audio.mp3 \
  --output-path predictions_r4q.csv
```

### Russell4Q Prediction (with track statistics)
```bash
python -m mer.modeling.predict \
  --model-path reports/training_<dataset>_Russell4Q_<timestamp>/model.pth \
  --audio-path audio.mp3 \
  --output-path predictions_r4q.csv \
  --return-average
```

### Russell4Q Prediction (detailed output)
```bash
python -m mer.modeling.predict \
  --model-path reports/training_<dataset>_Russell4Q_<timestamp>/model.pth \
  --audio-path audio.mp3 \
  --output-path predictions_r4q.csv \
  --return-average \
  --verbose
```

## Model Comparison

### DEAM
```bash
python -m mer.modeling.compare \
  --va-model reports/training_DEAM_VA_<timestamp>/model.pth \
  --russell4q-model reports/training_DEAM_Russell4Q_<timestamp>/model.pth \
  --dataset-name DEAM \
  --batch-size 8 \
  --test-size 0.1 \
  --seed 42 \
  --labels-scale norm \
  --output-dir reports/comparison_DEAM
```

### PMEmo
```bash
python -m mer.modeling.compare \
  --va-model reports/training_PMEmo_VA_<timestamp>/model.pth \
  --russell4q-model reports/training_PMEmo_Russell4Q_<timestamp>/model.pth \
  --dataset-name PMEmo \
  --batch-size 8 \
  --test-size 0.1 \
  --seed 42 \
  --labels-scale norm \
  --output-dir reports/comparison_PMEmo
```

### MERGE
```bash
python -m mer.modeling.compare \
  --va-model reports/training_MERGE_VA_<timestamp>/model.pth \
  --russell4q-model reports/training_MERGE_Russell4Q_<timestamp>/model.pth \
  --dataset-name MERGE \
  --batch-size 6 \
  --merge-split 70_15_15 \
  --output-dir reports/comparison_MERGE
```

## Streamlit Web App

```bash
streamlit run app.py
```

## Hyperparameter optimization
```bash
python -m mer.modeling.optimize \ 
  --dataset-name pmemo \ 
  --prediction-mode VA

python -m mer.modeling.optimize \ 
  --dataset-name merge \ 
  --prediction-mode Russell4Q 
```

## Tensorboard
```bash
tensorboard --logdir "reports"
```
