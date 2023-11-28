import torch


class SelfAdjDiceLoss(torch.nn.Module):
    r"""
    Creates a criterion that optimizes a multi-class Self-adjusting Dice Loss
    ("Dice Loss for Data-imbalanced NLP Tasks" paper)

    Args:
        alpha (float): a factor to push down the weight of easy examples
        gamma (float): a factor added to both the nominator and the denominator for smoothing purposes
        reduction (string): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed.

    Shape:
        - logits: `(N, C)` where `N` is the batch size and `C` is the number of classes.
        - targets: `(N)` where each value is in [0, C - 1]
    """

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

        probs = torch.sigmoid(logits)

        # Flatten label and prediction tensors
        probs = probs.view(-1)
        targets = targets.view(-1)
        intersection = (probs * targets).sum()

        dice = (2.0 * intersection + self.gamma) / (
            probs.sum() + targets.sum() + self.gamma
        )
        # probs = torch.softmax(logits, dim=1)
        # probs = torch.gather(probs, dim=1, index=targets.unsqueeze(1))

        # probs_with_factor = ((1 - probs) ** self.alpha) * probs
        # loss = 1 - (2 * probs_with_factor + self.gamma) / (
        #    probs_with_factor + 1 + self.gamma
        # )

        loss = 1 - dice

        print(loss)
        exit()
        # return loss
        return loss.mean()
