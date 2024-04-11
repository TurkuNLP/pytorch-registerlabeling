from pathlib import Path

from ray.tune.schedulers import ASHAScheduler
from ray.tune.schedulers import PopulationBasedTraining
from ray.tune import grid_search, CLIReporter, loguniform, choice, uniform
from ray.tune.search.hyperopt import HyperOptSearch
from ray import init as ray_init
from ..training_args import CustomTrainingArguments


def hyperparameter_search(
    trainer,
    hp_search_lib,
    working_dir,
    wandb_project_name,
    num_gpus,
    ray_log_path,
    min_lr,
    max_lr,
):
    absolute_path = str(Path(f"{working_dir}/{hp_search_lib}/").resolve())

    hp_config = {
        "direction": "maximize",
        "backend": hp_search_lib,
        "local_dir": absolute_path,
        "hp_space": {},
        "resources_per_trial": {
            "gpu": num_gpus,
        },
    }

    print(f"Ray is using {num_gpus} per trial")

    if hp_search_lib == "ray":
        ray_init(ignore_reinit_error=True, num_cpus=1, _temp_dir=ray_log_path)
        hp_config["scheduler"] = ASHAScheduler(metric="eval_f1", mode="max")
        hp_config["search_alg"] = HyperOptSearch(metric="eval_f1", mode="max")
        hp_config["hp_space"] = lambda _: {
            "learning_rate": loguniform(min_lr, max_lr),
            "per_device_train_batch_size": choice([x / num_gpus for x in [6, 8, 12]]),
            "loss_gamma": choice([1, 2, 3]),
            "loss_alpha": choice([0.5, 0.75, 0.25]),
        }

    best_model = trainer.hyperparameter_search(**hp_config)

    print(f"Best model according to {hp_search_lib}:")
    print(best_model)