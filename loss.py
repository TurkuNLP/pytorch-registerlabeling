import torch
import torch.nn.functional as F


class BCEFocalLoss(torch.nn.Module):
    def __init__(self, gamma=1.0, alpha=1.0, reduction="mean"):
        super(BCEFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        BCE_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")

        pt = torch.exp(-BCE_loss)

        loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        # Class balancing
        loss = loss * (targets * self.alpha + (1 - targets) * (1 - self.alpha))

        if self.reduction == "mean":
            return loss.mean()
        return loss


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
