import torch


class SelfAdjDiceLoss(torch.nn.Module):
    def __init__(
        self, alpha: float = 1.0, gamma: float = 1.0, reduction: str = "mean"
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
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
        # Apply sigmoid to the predictions
        predictions = torch.sigmoid(logits)

        # Calculate per-class weights
        class_weights = 1.0 / (targets.sum(dim=0) ** 2 + 1e-6)  # Avoid division by zero
        class_weights /= class_weights.sum()  # Normalize weights

        # Calculate intersection and union
        intersection = (predictions * targets).sum(dim=0)
        union = (predictions + targets).sum(dim=0)

        # Calculate weighted Dice coefficient
        dice = (
            2.0 * (intersection * class_weights) / (union * class_weights + self.gamma)
        )
        dice_loss = 1 - dice.sum() / len(dice)  # Average over classes

        return dice_loss

        preds = torch.sigmoid(logits)
        num_labels = preds.shape[1]

        # Calculate label weights inversely proportional to label frequency
        label_volumes = (
            targets.sum(dim=0) + 1e-6
        )  # Add small constant to avoid division by zero
        label_weights = 1.0 / label_volumes

        weighted_preds = preds * label_weights
        weighted_targets = targets * label_weights

        # Flatten label and prediction tensors
        probs = weighted_preds.view(-1)
        targets = weighted_targets.view(-1)

        intersection = (probs * targets).sum()
        dice = (2.0 * intersection + self.gamma) / (
            probs.sum() + targets.sum() + self.gamma
        )

        # Calculate average Dice coefficient
        average_dice = dice / num_labels

        # Return negative average Dice for loss
        return -average_dice

        """
        preds = torch.sigmoid(logits)

        num_cols = preds.shape[1]
        dice = 0
        for i in range(num_cols):
            pred_col = preds[:, i]
            target_col = targets[:, i]
            dice += self.dice_coeff(pred_col, target_col)

        average_dice = dice / num_cols

        return 1 - average_dice

        print(dice)

        """
        probs = torch.sigmoid(logits)

        # Flatten label and prediction tensors
        probs = probs.view(-1)
        targets = targets.view(-1)
        intersection = (probs * targets).sum()

        dice = (2.0 * intersection + self.gamma) / (
            probs.sum() + targets.sum() + self.gamma
        )

        dice = dice / torch.sigmoid(logits).shape[1]
        # probs = torch.softmax(logits, dim=1)
        # probs = torch.gather(probs, dim=1, index=targets.unsqueeze(1))

        # probs_with_factor = ((1 - probs) ** self.alpha) * probs
        # loss = 1 - (2 * probs_with_factor + self.gamma) / (
        #    probs_with_factor + 1 + self.gamma
        # )

        loss = 1 - dice

        return loss

        print(loss)
        exit()
        # return loss
        return loss.mean()
