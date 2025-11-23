from typing import Callable

import torch


def masked_mse(pred, target, mask):
    mask = mask.unsqueeze(-1)
    num = ((pred - target) ** 2 * mask).sum()
    den = mask.sum() * pred.shape[-1]
    return num / (den + 1e-8)


def masked_ccc_loss(pred, target, mask):
    eps = 1e-8
    m = mask.unsqueeze(-1)
    valid = m > 0
    if valid.sum() == 0:
        return torch.tensor(0.0, device=pred.device, requires_grad=True)
    losses = []
    for d in range(pred.shape[-1]):
        x = pred[..., d][valid[..., 0]]
        y = target[..., d][valid[..., 0]]
        mx = torch.mean(x)
        my = torch.mean(y)
        vx = torch.var(x, unbiased=False)
        vy = torch.var(y, unbiased=False)
        cov = torch.mean((x - mx) * (y - my))
        ccc = (2 * cov) / (vx + vy + (mx - my) ** 2 + eps)
        losses.append(1.0 - ccc)
    return torch.mean(torch.stack(losses))


def masked_hybrid(pred, target, mask):
    return (masked_mse(pred, target, mask) + masked_ccc_loss(pred, target, mask)) / 2


def masked_cross_entropy(pred, target, mask):
    """
    Masked cross-entropy loss for classification.
    Args:
        pred: (batch_size, seq_len, num_classes) logits
        target: (batch_size, seq_len) class indices
        mask: (batch_size, seq_len) binary mask
    Returns:
        Scalar loss
    """
    # Flatten to (batch_size * seq_len, num_classes) and (batch_size * seq_len,)
    batch_size, seq_len, num_classes = pred.shape
    pred_flat = pred.reshape(-1, num_classes)
    target_flat = target.reshape(-1).long()
    mask_flat = mask.reshape(-1)

    # Compute cross-entropy loss
    loss = torch.nn.functional.cross_entropy(pred_flat, target_flat, reduction="none")

    # Apply mask and average
    masked_loss = loss * mask_flat
    return masked_loss.sum() / (mask_flat.sum() + 1e-8)


def make_loss_fn(loss_type: str) -> Callable:
    loss_type = str(loss_type).lower()
    if loss_type == "masked_mse":
        return masked_mse
    elif loss_type == "masked_ccc":
        return masked_ccc_loss
    elif loss_type == "masked_hybrid":
        return masked_hybrid
    elif loss_type == "masked_cross_entropy":
        return masked_cross_entropy
    else:
        raise NotImplementedError(loss_type)
