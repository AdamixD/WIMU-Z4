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


def make_loss_fn(loss_type: str) -> Callable:
    lt = str(loss_type).lower()
    match lt:
        case 'masked_mse':
            return masked_mse
        case 'masked_ccc':
            return masked_ccc_loss
        case 'masked_hybrid':
            return masked_hybrid
        case _:
            raise NotImplemented(loss_type)
