import torch
import torch.nn.functional as F


# BCE Focal Loss
def BCEFocalLoss(outputs, labels, loss_gamma, loss_alpha):
    logits = outputs.logits
    BCE_loss = F.binary_cross_entropy_with_logits(
        logits, labels.float(), reduction="none"
    )
    pt = torch.exp(-BCE_loss)
    loss = loss_alpha * (1 - pt) ** loss_gamma * BCE_loss

    # Class balancing
    loss = loss * (labels * loss_alpha + (1 - labels) * (1 - loss_alpha))
    loss = loss.mean()

    return loss
