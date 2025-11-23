import numpy as np


def flatten(x):
    return np.asarray(x, dtype=float).ravel()


def pearson(y, p):
    y = flatten(y)
    p = flatten(p)
    if y.size < 2:
        return float("nan")
    return float(np.corrcoef(y, p)[0, 1])


def rmse(y, p):
    y = flatten(y)
    p = flatten(p)
    if y.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean((y - p) ** 2)))


def r2(y, p):
    y = flatten(y)
    p = flatten(p)
    if y.size < 2:
        return float("nan")
    ss_res = np.sum((y - p) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    if ss_tot == 0:
        return float("nan")
    return float(1.0 - ss_res / ss_tot)


def ccc(y, p):
    y = flatten(y)
    p = flatten(p)
    if y.size < 2:
        return float("nan")
    my, mp = np.mean(y), np.mean(p)
    vy, vp = np.var(y), np.var(p)
    cov = np.mean((y - my) * (p - mp))
    denom = vy + vp + (my - mp) ** 2
    if denom == 0:
        return float("nan")
    return float(2 * cov / denom)


def metrics_dict(y, p):
    return {
        "CCC": ccc(y, p),
        "Pearson": pearson(y, p),
        "R2": r2(y, p),
        "RMSE": rmse(y, p),
    }


def labels_convert(y, src: str, dst: str):
    y = np.asarray(y, dtype=float)
    if src == dst:
        return y
    if src == "19" and dst == "norm":
        return (y - 5.0) / 4.0
    if src == "norm" and dst == "19":
        return 4.0 * y + 5.0
    raise ValueError(f"Unsupported scale conversion {src} -> {dst}")


def classification_metrics(y_true, y_pred):
    """
    Compute classification metrics.
    
    Args:
        y_true: Ground truth class labels (N,)
        y_pred: Predicted class labels (N,)
    
    Returns:
        Dictionary with accuracy, precision, recall, F1 score, and confusion matrix
    """
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
    )

    y_true = flatten(y_true).astype(int)
    y_pred = flatten(y_pred).astype(int)

    if y_true.size == 0:
        return {
            "Accuracy": float("nan"),
            "Precision": float("nan"),
            "Recall": float("nan"),
            "F1": float("nan"),
            "Confusion_Matrix": np.array([]),
        }

    # Determine number of classes
    num_classes = max(y_true.max(), y_pred.max()) + 1

    return {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Precision": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "F1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "Confusion_Matrix": confusion_matrix(y_true, y_pred, labels=np.arange(num_classes)),
    }


def va_to_russell_quadrant(valence, arousal):
    """
    Map VA values to Russell's 4 quadrants.
    
    Russell's Circumplex Model:
    - Q1: High Arousal (>0), High Valence (>0) - Happy/Excited
    - Q2: High Arousal (>0), Low Valence (<=0) - Angry/Tense  
    - Q3: Low Arousal (<=0), Low Valence (<=0) - Sad/Depressed
    - Q4: Low Arousal (<=0), High Valence (>0) - Calm/Relaxed
    
    Args:
        valence: Valence values in range [-1, 1]
        arousal: Arousal values in range [-1, 1]
    
    Returns:
        Quadrant indices (0=Q1, 1=Q2, 2=Q3, 3=Q4)
    """
    valence = np.asarray(valence)
    arousal = np.asarray(arousal)

    # Initialize quadrant array
    quadrants = np.zeros(valence.shape, dtype=int)

    # Q1: High arousal, High valence
    quadrants[(arousal > 0) & (valence > 0)] = 0

    # Q2: High arousal, Low valence
    quadrants[(arousal > 0) & (valence <= 0)] = 1

    # Q3: Low arousal, Low valence
    quadrants[(arousal <= 0) & (valence <= 0)] = 2

    # Q4: Low arousal, High valence
    quadrants[(arousal <= 0) & (valence > 0)] = 3

    return quadrants


def quadrant_to_name(quadrant_idx):
    """
    Convert quadrant index to descriptive name.
    
    Args:
        quadrant_idx: Integer or array of integers (0-3)
    
    Returns:
        String or array of strings with quadrant names
    """
    mapping = {
        0: "Q1 (Happy/Excited)",
        1: "Q2 (Angry/Tense)",
        2: "Q3 (Sad/Depressed)",
        3: "Q4 (Calm/Relaxed)",
    }

    if isinstance(quadrant_idx, (int, np.integer)):
        return mapping.get(int(quadrant_idx), "Unknown")
    else:
        return np.array([mapping.get(int(q), "Unknown") for q in quadrant_idx])


# Alias for backward compatibility
va_to_russell4q = va_to_russell_quadrant
