import torch


def linear_warmup_decay(warmup_steps, total_steps, cosine=False):
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        if cosine:
            progress = float(current_step - warmup_steps) / float(
                max(1, total_steps - warmup_steps)
            )
            return 0.5 * (1.0 + torch.cos(torch.pi * progress))
        else:
            return 1.0 - max(
                0,
                float(current_step - warmup_steps) / float(total_steps - warmup_steps),
            )

    return lr_lambda
