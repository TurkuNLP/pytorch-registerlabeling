import torch


class SelfAdjustingMultiLabelDiceLoss(torch.nn.Module):
    def __init__(self, gamma=1.0, alpha=1.0):
        super(SelfAdjustingMultiLabelDiceLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

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
