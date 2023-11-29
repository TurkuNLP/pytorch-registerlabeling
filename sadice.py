import torch
import torch.nn.functional as F


class SelfAdjustingMultiLabelDiceLoss(torch.nn.Module):
    def __init__(self, gamma=1.0, alpha=1.0, reduction="mean"):
        super(SelfAdjustingMultiLabelDiceLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def dice_coeff(self, y_true, y_pred):
        y_true_f = torch.flatten(y_true)
        y_pred_f = torch.flatten(y_pred)
        intersection = torch.sum(y_true_f * y_pred_f)
        dice = (2.0 * intersection + self.gamma) / (
            torch.sum(y_true_f) + torch.sum(y_pred_f) + self.gamma
        )
        return dice

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Calculate probabilities
        probs = torch.sigmoid(logits)
        pt = probs * targets + (1 - probs) * (1 - targets)

        # Calculate the focal loss component
        focal_weight = (1 - pt).pow(self.gamma)

        # Compute BCE loss
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")

        # Apply focal loss weighting
        loss = self.alpha * focal_weight * bce_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss

        preds = torch.sigmoid(logits)
        num_labels = preds.shape[1]
        print(num_labels)

        total_dice = 0
        for i in range(num_labels):
            pred_col = preds[:, i]
            target_col = targets[:, i]
            # Apply the alpha factor
            error = (1 - pred_col) ** self.alpha
            weighted_pred_col = error * pred_col
            # Compute dice coefficient for this label
            dice = self.dice_coeff(target_col, weighted_pred_col)
            total_dice += dice

        # Calculate average Dice coefficient
        average_dice = total_dice / num_labels

        # Return negative average Dice for loss
        return 1 - average_dice
