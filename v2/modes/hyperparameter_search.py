from ray.tune.schedulers import ASHAScheduler
from ray.tune.schedulers import PopulationBasedTraining
from ray.tune import grid_search, CLIReporter, loguniform, choice, uniform
from ray.tune.search.hyperopt import HyperOptSearch
from ray import init as ray_init


def hyperparameter_search(trainer, hp_search_lib, working_dir, wandb_project_name):
    hp_config = {
        "direction": "maximize",
        "backend": hp_search_lib,
        "local_dir": f"{working_dir}/{hp_search_lib}",
        "hp_space": {},
        "n_trials": 1,
    }

    if hp_search_lib == "ray":
        ray_init(ignore_reinit_error=True, num_cpus=1)
        hp_config["scheduler"] = ASHAScheduler(metric="eval_f1", mode="max")
        # hp_config["search_alg"] = HyperOptSearch(metric="eval_f1", mode="max")
        hp_config["hp_space"] = lambda _: {
            "learning_rate": [1e-6, 5e-6, 1e-5, 5e-5, 1e-4],
            "per_device_train_batch_size": [6, 8, 12],
        }

    elif hp_search_lib == "wandb":
        hp_config["hp_space"] = lambda _: {
            "method": "bayes",
            "name": wandb_project_name,
            "metric": {"goal": "maximize", "name": "eval_f1"},
            "parameters": {
                "learning_rate": {
                    "distribution": "uniform",
                    "min": 1e-6,
                    "max": 1e-4,
                },
                "per_device_train_batch_size": {"values": [6, 8, 12, 16]},
            },
        }

    best_model = trainer.hyperparameter_search(**hp_config)

    print(f"Best model according to {hp_search_lib}:")
    print(best_model)
