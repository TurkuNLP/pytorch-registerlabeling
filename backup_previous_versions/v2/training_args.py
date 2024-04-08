from transformers import TrainingArguments


# Extend TrainingArguments to include custom hyperparameters
class CustomTrainingArguments(TrainingArguments):
    def __init__(
        self,
        loss_gamma,
        loss_alpha,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.loss_gamma = loss_gamma
        self.loss_alpha = loss_alpha
