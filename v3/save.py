import csv
import json
import os
import shutil
import tempfile

import torch

from .labels import decode_binary_labels


def save_checkpoint(
    cfg, model, optimizer, lr_scheduler, scaler, dev_metrics, custom_model
):
    checkpoint_dir = f"{cfg.working_dir}/best_checkpoint"
    os.makedirs(cfg.working_dir, exist_ok=True)
    if not custom_model:
        if cfg.gpus > 1:
            model.module.save_pretrained(checkpoint_dir)
        else:
            model.save_pretrained(checkpoint_dir)
    else:
        os.makedirs(checkpoint_dir, exist_ok=True)
        torch.save(model.state_dict(), f"{checkpoint_dir}/model_state.pth")
        model.config.to_json_file(f"{checkpoint_dir}/config.json")
    torch.save(optimizer.state_dict(), f"{checkpoint_dir}/optimizer_state.pth")
    torch.save(lr_scheduler.state_dict(), f"{checkpoint_dir}/lr_scheduler_state.pth")
    if cfg.use_amp:
        torch.save(scaler.state_dict(), f"{checkpoint_dir}/scaler_state.pth")

    with open(f"{checkpoint_dir}/dev_metrics.json", "w") as f:
        json.dump(dev_metrics, f)


def save_model(working_dir):
    shutil.rmtree(f"{working_dir}/best_model", ignore_errors=True)
    shutil.copytree(
        f"{working_dir}/best_checkpoint",
        f"{working_dir}/best_model",
    )


def save_predictions(trues, preds, metrics, cfg):
    true_labels_str = decode_binary_labels(trues, cfg.label_scheme)
    predicted_labels_str = decode_binary_labels(preds, cfg.label_scheme)

    data = list(zip(true_labels_str, predicted_labels_str))
    out_file = f"{cfg.working_dir}/test_predictions_{cfg.data.test or cfg.data.dev or cfg.data.train}_{cfg.trainer.learning_rate}.csv"
    out_file_metrics = f"{cfg.working_dir}/test_metrics_{cfg.data.test or cfg.data.dev or cfg.data.train}_{cfg.trainer.learning_rate}.json"

    with open(out_file, "w", newline="") as csvfile:
        csv_writer = csv.writer(csvfile, delimiter="\t")
        csv_writer.writerows(data)

    with open(out_file_metrics, "w") as f:
        json.dump(metrics, f)

    print(f"Predictions saved to {out_file}")
    print(f"Metrics saved to {out_file_metrics}")


def save_ray_checkpoint(model, optimizer, train, dev_metrics):
    with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
        path = os.path.join(temp_checkpoint_dir, "checkpoint.pt")
        torch.save((model.state_dict(), optimizer.state_dict()), path)
        checkpoint = train.Checkpoint.from_directory(temp_checkpoint_dir)
        train.report(
            {
                "loss": dev_metrics["eval_loss"],
                "f1": dev_metrics["eval_f1"],
            },
            checkpoint=checkpoint,
        )


def init_ray_dir(path):
    shutil.rmtree(path, ignore_errors=True)
    os.makedirs(path, exist_ok=True)
