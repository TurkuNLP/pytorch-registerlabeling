import torch
import torch.nn.functional as F


# BCE Focal Loss
def BCEFocalLoss(outputs, labels, loss_gamma, loss_alpha, num_labels):
    BCE_loss = F.binary_cross_entropy_with_logits(
        outputs.logits,
        labels.float(),
        reduction="none",
    )
    print(BCE_loss)
    exit()
    pt = torch.exp(-BCE_loss)
    loss = loss_alpha * (1 - pt) ** loss_gamma * BCE_loss

    # Class balancing
    loss = loss * (labels * loss_alpha + (1 - labels) * (1 - loss_alpha))
    loss = loss.mean()

    return loss


class BCEClassFocalLoss(torch.nn.Module):
    def __init__(self, gamma=1.0, alpha=1.0):
        super(BCEFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        BCE_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        pt = torch.exp(-BCE_loss)
        loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        # Class balancing
        loss = loss * (targets * self.alpha + (1 - targets) * (1 - self.alpha))

        if self.reduction == "mean":
            return loss.mean()
        return loss
