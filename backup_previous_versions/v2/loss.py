import torch
import torch.nn.functional as F

from .labels import label_hierarchy


class HierarchicalBCEFocalLoss(torch.nn.Module):
    def __init__(
        self,
        gamma=1.0,
        alpha=1.0,
        threshold=0.5,
        hierarchy_penalty_weight=1.0,
        reduction="mean",
    ):
        super(HierarchicalBCEFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.hierarchy_penalty_weight = hierarchy_penalty_weight
        self.label_hierarchy = (
            label_hierarchy  # A mapping of child labels to parent labels
        )
        self.threshold = threshold  # Dynamic threshold for predictions

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Compute standard BCE Focal Loss
        BCE_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        pt = torch.exp(-BCE_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        # Compute Hierarchical Penalty
        hierarchy_penalty = torch.zeros_like(focal_loss)
        for child, parent in self.label_hierarchy.items():
            child_pred = logits[:, child]
            parent_target = targets[:, parent]
            # Apply penalty when child is predicted but parent is not, using dynamic threshold
            penalty_condition = (child_pred > self.threshold) & (
                parent_target < self.threshold
            )
            hierarchy_penalty[:, child] += penalty_condition * focal_loss[:, child]

        # Combine losses
        combined_loss = focal_loss + self.hierarchy_penalty_weight * hierarchy_penalty

        # Class balancing
        combined_loss = combined_loss * (
            targets * self.alpha + (1 - targets) * (1 - self.alpha)
        )

        if self.reduction == "mean":
            return combined_loss.mean()
        return combined_loss


class BCEFocalLoss(torch.nn.Module):
    def __init__(self, gamma=1.0, alpha=1.0, reduction="mean"):
        super(BCEFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
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
