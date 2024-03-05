import json
import random

import numpy as np
import torch
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from ray import init as ray_init
from ray import train, tune
from ray.air.integrations.wandb import WandbLoggerCallback
from ray.tune.search.hyperopt import HyperOptSearch
from ray.util import inspect_serializability
from sentence_transformers import SentenceTransformer
from torch.nn.parallel import DataParallel
from torch.optim.lr_scheduler import LambdaLR
from tqdm import trange, tqdm
from transformers import (
    AutoModel,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
)

import wandb

from .data import get_dataset, preprocess_data
from .dataloader import init_dataloaders
from .embeddings import extract_doc_embeddings, extract_st_doc_embeddings
from .keywords import extract_keywords, analyze_keywords
from .labels import get_label_scheme
from .loss import BCEFocalLoss
from .metrics import compute_metrics
from .model import (
    LogisticRegressionModel,
    PooledRobertaForSequenceClassification,
    SentenceTransformerClassifier,
)
from .optimizer import create_optimizer
from .save import (
    init_ray_dir,
    save_checkpoint,
    save_model,
    save_predictions,
    save_ray_checkpoint,
)
from .scheduler import linear_warmup_decay
from .setfit_trainer import setfit_train
from .utils import (
    convert_embeddings_to_input,
    format_working_dir,
    get_eval_step,
    get_linear_modules,
    get_torch_dtype,
    log_gpu_memory,
    model_has_improved,
    model_save_condition,
)


class Main:
    def __init__(self, cfg):
        cfg.torch_dtype_torch = get_torch_dtype(cfg.torch_dtype)
        cfg.label_scheme = get_label_scheme(cfg.data.labels)
        cfg.num_labels = len(cfg.label_scheme)
        cfg.device_str = cfg.device
        cfg.device = torch.device(cfg.device)
        cfg.working_dir = format_working_dir(cfg.model.name, cfg.data, cfg.seed)
        cfg.working_dir_root = "/".join(cfg.working_dir.split("/")[:-1])
        cfg.wandb_project = cfg.working_dir.split("/", 1)[1].replace("/", ",")
        self.cfg = cfg
        print(f"Predicting {cfg.num_labels} labels")
        print(f"Working directory: {self.cfg.working_dir}")

        # Tf32
        if not self.cfg.no_tf32:
            torch.set_float32_matmul_precision("high")
            torch.backends.cudnn.allow_tf32

        # Make process deterministic
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)
        random.seed(cfg.seed)
        torch.cuda.manual_seed_all(cfg.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        if not cfg.no_data:

            # Init tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                cfg.model.name,
                torch_dtype=cfg.torch_dtype if not cfg.use_amp else torch.float32,
            )

            if cfg.set_pad_token:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # Prepare dataset
            self.dataset = preprocess_data(get_dataset(cfg), self.tokenizer, cfg)

            # Init dataloaders
            self.dataloaders = init_dataloaders(
                self.dataset, cfg.dataloader, self.tokenizer.pad_token_id, cfg.device
            )

            torch.set_default_device(self.cfg.device)

        # Run
        getattr(self, cfg.method)()

    def _wrap_peft(self):
        print("Wrapping PEFT model")

        if self.cfg.peft.target_modules == "linear":
            target_modules = get_linear_modules(self.model)
        else:
            target_modules = self.cfg.peft.target_modules.split(",")

        self.lora_config = LoraConfig(
            r=self.cfg.peft.lora_rank,
            lora_alpha=self.cfg.peft.lora_alpha,
            target_modules=target_modules,
            lora_dropout=0.05,
            task_type=TaskType.SEQ_CLS,
            use_dora=True,
        )
        self.model = prepare_model_for_kbit_training(self.model)
        self.model = get_peft_model(self.model, self.lora_config)
        self.model.print_trainable_parameters()

    def _init_model(self, model_path=None):
        model_params = {
            "num_labels": self.cfg.num_labels,
        }

        model_cls = AutoModelForSequenceClassification
        if self.cfg.train_using_embeddings:
            model_cls = AutoModel
        if self.cfg.model.roberta_pooled:
            model_cls = PooledRobertaForSequenceClassification
            model_params["pooling"] = self.cfg.model.roberta_pooled
        if self.cfg.model.low_cpu_mem_usage:
            model_params["low_cpu_mem_usage"] = True
        if self.cfg.model.quantize:
            model_params["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )

        if not (self.cfg.model.quantize or self.cfg.use_amp):
            model_params["torch_dtype"] = self.cfg.torch_dtype_torch

        if self.cfg.use_fa2:
            model_params["attn_implementation"] = "flash_attention_2"

        if self.cfg.train_using_embeddings:
            self.classification_head = LogisticRegressionModel(
                input_size=self.cfg.train_using_embeddings,
                num_labels=self.cfg.num_labels,
                torch_dtype=self.cfg.torch_dtype_torch,
            ).to(self.cfg.torch_dtype_torch)

        model = model_cls.from_pretrained(
            self.cfg.model.name if not model_path else model_path, **model_params
        )

        if self.cfg.gpus > 1:
            model = DataParallel(model, device_ids=list(range(self.cfg.gpus)))

        if not self.cfg.model.quantize:
            model = model.to(
                self.cfg.device,
                dtype=(
                    self.cfg.torch_dtype_torch
                    if not self.cfg.use_amp
                    else torch.float32
                ),
            )

        self.model = model

        if self.cfg.temp_test:
            self.model = SentenceTransformerClassifier(model, self.cfg.num_labels)

    def _train(self, config={}):
        self._init_model(
            self.cfg.resume if (self.cfg.resume and not self.cfg.peft.enable) else None
        )

        if self.cfg.peft.enable:
            if not self.cfg.resume:
                self._wrap_peft()
            else:
                self.model.load_adapter(self.cfg.resume)

        if self.cfg.set_pad_token:
            self.model.config.pad_token_id = self.tokenizer.pad_token_id

        num_training_steps = int(
            self.cfg.trainer.epochs
            * len(self.dataloaders["train"])
            / self.cfg.trainer.gradient_accumulation_steps
        )

        self.optimizer = create_optimizer(
            self.model,
            {
                "lr": config.get("learning_rate", self.cfg.trainer.learning_rate),
                "weight_decay": config.get(
                    "weight_decay", self.cfg.trainer.weight_decay
                ),
            },
        )

        self.scaler = torch.cuda.amp.GradScaler(enabled=self.cfg.use_amp)

        if self.cfg.resume:
            self.optimizer.load_state_dict(
                torch.load(f"{self.cfg.resume}/optimizer_state.pth")
            )

            if self.cfg.use_amp:
                self.scaler.load_state_dict(
                    torch.load(f"{self.cfg.resume}/scaler_state.pth")
                )

        self.lr_scheduler = LambdaLR(
            self.optimizer,
            linear_warmup_decay(
                # math.ceil(num_training_steps * self.cfg.trainer.warmup_ratio),
                num_training_steps * self.cfg.trainer.warmup_ratio,
                num_training_steps,
            ),
        )

        best_starting_score = False

        if self.cfg.resume:
            self.lr_scheduler.load_state_dict(
                torch.load(f"{self.cfg.resume}/lr_scheduler_state.pth")
            )
            with open(f"{self.cfg.resume}/model_state.json", "r") as f:
                loaded_data = json.load(f)
                best_starting_score = loaded_data[self.cfg.trainer.best_model_metric]
                print(
                    f"Previous best {self.cfg.trainer.best_model_metric} was {best_score}"
                )

        progress_bar = tqdm(
            range(num_training_steps), mininterval=self.cfg.tqdm_mininterval
        )
        best_score = best_starting_score

        ##### TRAINING LOOP STARTS HERE
        self.model.train()
        epoch = 0
        batch_i = 0
        remaining_patience = self.cfg.trainer.patience
        running_loss = 0
        eval_step = get_eval_step(
            len(self.dataloaders["train"]), self.cfg.trainer.eval_step
        )
        while remaining_patience > 0:
            epoch += 1
            batch_losses = []
            for batch in self.dataloaders["train"]:
                batch_i += 1

                batch = {k: v.to(self.cfg.device) for k, v in batch.items()}
                labels = batch.pop("labels")

                with torch.autocast(
                    device_type=self.cfg.device_str,
                    dtype=self.cfg.torch_dtype_torch,
                    enabled=self.cfg.use_amp,
                ):

                    outputs = self.model(**batch)

                    if self.cfg.train_using_embeddings:
                        outputs = self.classification_head(
                            **convert_embeddings_to_input(outputs, batch)
                        )

                    # if type(outputs) is tuple:
                    #    outputs = outputs[0]

                    loss = BCEFocalLoss(
                        outputs,
                        labels,
                        self.cfg.trainer.loss_gamma,
                        self.cfg.trainer.loss_alpha,
                    )

                batch_losses.append(loss.item())
                loss = loss / self.cfg.trainer.gradient_accumulation_steps
                self.scaler.scale(loss).backward()
                if batch_i % self.cfg.trainer.gradient_accumulation_steps == 0:
                    if self.cfg.trainer.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.cfg.trainer.max_grad_norm,
                        )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)

                    progress_bar.update(1)
                    progress_bar.set_description(
                        f"E-{epoch}:{int((batch_i % eval_step)/eval_step * 100)}% ({remaining_patience}/{self.cfg.trainer.patience}), loss: {(sum(batch_losses) / len(batch_losses)):4f}",
                        refresh=False,
                    )

                    running_loss = sum(batch_losses) / len(batch_losses)

                if batch_i % eval_step == 0:
                    print(f"Loss at step {batch_i} [E-{epoch}]: {running_loss}")
                    dev_metrics = self._evaluate()
                    self.model.train()
                    print(dev_metrics)
                    if not self.cfg.method == "ray_tune":
                        wandb.log(
                            {
                                **dev_metrics,
                                **{
                                    "train_loss": running_loss,
                                    "learning_rate": self.optimizer.param_groups[0][
                                        "lr"
                                    ],
                                    "step": batch_i + 1,
                                    "epoch": epoch,
                                },
                            }
                        )

                    else:
                        save_ray_checkpoint(
                            self.model, self.optimizer, train, dev_metrics
                        )

                    patience_metric = dev_metrics[self.cfg.trainer.best_model_metric]

                    if best_score is False or model_has_improved(
                        self.cfg.trainer.best_model_metric, patience_metric, best_score
                    ):
                        best_score = patience_metric
                        save_checkpoint(
                            self.cfg,
                            self.model,
                            self.optimizer,
                            self.lr_scheduler,
                            self.scaler,
                            dev_metrics,
                        )
                        remaining_patience = self.cfg.trainer.patience
                    else:
                        remaining_patience -= 1

        do_save = model_save_condition(self.cfg, best_score, best_starting_score)

        if do_save:
            save_model(self.cfg.working_dir)
            print(f"Model saved to {self.cfg.working_dir}")

        if self.cfg.predict:
            self.predict(from_checkpoint=not do_save)

    def _evaluate(self, split="dev", timer=False):
        self.model.eval()
        batch_logits = []
        batch_labels = []
        batch_losses = []
        data_len = len(self.dataloaders[split])

        progress_bar = tqdm(range(data_len))
        progress_bar.set_description(f"Evaluating {split} split")

        if timer:
            starter = torch.cuda.Event(enable_timing=True)
            ender = torch.cuda.Event(enable_timing=True)
            timings = np.zeros(data_len)

        for batch_i, batch in enumerate(self.dataloaders[split]):
            batch = {k: v.to(self.cfg.device) for k, v in batch.items()}
            labels = batch.pop("labels")
            with torch.no_grad():
                if timer:
                    starter.record()
                outputs = self.model(**batch)

                if self.cfg.train_using_embeddings:
                    outputs = self.classification_head(
                        **convert_embeddings_to_input(outputs, batch)
                    )
                if timer:
                    ender.record()
                    torch.cuda.synchronize()
                    curr_time = starter.elapsed_time(ender)
                    timings[batch_i] = curr_time / len(batch["input_ids"])

            loss = BCEFocalLoss(
                outputs,
                labels,
                self.cfg.trainer.loss_gamma,
                self.cfg.trainer.loss_alpha,
            )
            batch_logits.append(outputs.logits)
            batch_labels.append(labels)
            batch_losses.append(loss.item())

            progress_bar.update(1)

        progress_bar.close()

        if timer:
            mean_syn = np.mean(timings)
            std_syn = np.std(timings)
            print(f"Avg. instance inference time: {mean_syn:4f} ({std_syn:4f})")

        metrics = compute_metrics(
            torch.cat(batch_logits, dim=0),
            torch.cat(batch_labels, dim=0),
            split,
            self.cfg.label_scheme,
        )
        if split == "dev":
            metrics["eval_loss"] = sum(batch_losses) / len(batch_losses)
            return metrics

        elif split == "test":
            if self.cfg.save_predictions:
                save_predictions(*metrics[1], metrics[0], self.cfg)
            return metrics[0]

    def predict(self, from_checkpoint=False):
        model_path = f"{self.cfg.working_dir}/best_{'checkpoint' if from_checkpoint or self.cfg.predict_from_checkpoint else 'model'}"

        if self.cfg.peft.enable:
            self._init_model()
            self.model.load_adapter(model_path)
        else:
            self._init_model(model_path)

        if self.cfg.data.dev or self.cfg.method == "finetune":
            print("Final dev set evaluation")
            print(self._evaluate())

        print("Test set evaluation")
        print(self._evaluate("test", timer=self.cfg.timer))

    def finetune(self):
        wandb.login()
        wandb.init(
            project=f"finetune_{self.cfg.wandb_project[:100]}",
            config=self.cfg,
        )
        self._train()

    def ray_tune(self):

        wandb.login()
        self.cfg.tqdm_mininterval = 10
        self.cfg.model.save = False
        self.cfg.predict = False

        config = {
            "learning_rate": tune.quniform(
                *self.cfg.ray.learning_rate, self.cfg.ray.learning_rate[0]
            ),
            "loss_gamma": tune.uniform(*self.cfg.ray.loss_gamma),
            "loss_alpha": tune.uniform(*self.cfg.ray.loss_alpha),
        }
        scheduler = tune.schedulers.ASHAScheduler()

        ray_init(
            ignore_reinit_error=True, num_cpus=1, _temp_dir=self.cfg.root_path + "/tmp"
        )
        ray_dir = f"{self.cfg.root_path}/tmp/ray/{self.cfg.wandb_project}"
        init_ray_dir(ray_dir)

        tuner = tune.Tuner(
            tune.with_resources(
                tune.with_parameters(self._train),
                resources={"cpu": 1, "gpu": self.cfg.ray.gpus_per_trial},
            ),
            tune_config=tune.TuneConfig(
                metric="loss",
                mode="min",
                scheduler=scheduler,
                num_samples=20,
                search_alg=HyperOptSearch(metric="loss", mode="min"),
            ),
            run_config=train.RunConfig(
                name=self.cfg.wandb_project,
                callbacks=[WandbLoggerCallback(project=self.cfg.wandb_project)],
                storage_path=ray_dir,
                local_dir=ray_dir,
            ),
            param_space=config,
        )
        results = tuner.fit()

        best_result = results.get_best_result("loss", "min")

        print("Best trial config: {}".format(best_result.config))
        print(
            "Best trial final validation loss: {}".format(best_result.metrics["loss"])
        )
        print("Best trial final validation f1: {}".format(best_result.metrics["f1"]))

    def hf_finetune(self):
        from transformers import (
            DataCollatorWithPadding,
            EarlyStoppingCallback,
            Trainer,
            TrainingArguments,
        )

        wandb.login()
        self._init_model()

        loss_gamma = self.cfg.trainer.loss_gamma
        loss_alpha = self.cfg.trainer.loss_alpha

        def compute_metrics_fn(p):
            _, labels = p
            predictions = (
                p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
            )
            return compute_metrics(predictions, labels, "dev", self.cfg.label_scheme)

        class MultiLabelTrainer(Trainer):
            def __init__(self, *args, **kwargs):
                super(MultiLabelTrainer, self).__init__(*args, **kwargs)

            def compute_loss(self, model, inputs, return_outputs=False):
                labels = inputs.pop("labels")
                outputs = model(**inputs)
                loss = BCEFocalLoss(outputs, labels, loss_gamma, loss_alpha)

                return (loss, outputs) if return_outputs else loss

        trainer = MultiLabelTrainer(
            model=self.model,
            args=TrainingArguments(
                f"{self.cfg.working_dir}/hf_checkpoints",
                seed=self.cfg.seed,
                overwrite_output_dir=not self.cfg.resume,
                evaluation_strategy="epoch",
                save_strategy="epoch",
                logging_strategy="epoch",
                load_best_model_at_end=self.cfg.model.save,
                save_total_limit=2,
                weight_decay=self.cfg.trainer.weight_decay,
                warmup_ratio=self.cfg.trainer.warmup_ratio,
                learning_rate=self.cfg.trainer.learning_rate,
                max_grad_norm=self.cfg.trainer.max_grad_norm,
                lr_scheduler_type="linear",
                metric_for_best_model=self.cfg.trainer.best_model_metric,
                greater_is_better=not "loss" in self.cfg.trainer.best_model_metric,
                per_device_train_batch_size=self.cfg.dataloader.train_batch_size,
                per_device_eval_batch_size=self.cfg.dataloader.dev_batch_size,
                num_train_epochs=self.cfg.trainer.epochs,
                gradient_checkpointing=False,
                gradient_accumulation_steps=self.cfg.trainer.gradient_accumulation_steps,
                optim="adamw_torch",
                bf16=self.cfg.bf16,
                tf32=not self.cfg.no_tf32,
                resume_from_checkpoint=self.cfg.resume,
            ),
            train_dataset=self.dataset.get("train", []),
            eval_dataset=self.dataset.get("dev", []),
            compute_metrics=compute_metrics_fn,
            tokenizer=self.tokenizer,
            data_collator=DataCollatorWithPadding(
                tokenizer=self.tokenizer,
                padding="longest",
                max_length=self.cfg.data.max_length,
            ),
            callbacks=[
                EarlyStoppingCallback(early_stopping_patience=self.cfg.trainer.patience)
            ],
        )

        trainer.train()

    def extract_doc_embeddings(self):
        path = self.cfg.working_dir_root + "/embeddings"
        self._init_model()
        extract_doc_embeddings(
            self.model, self.dataset, path, self.cfg.device, self.cfg.embeddings
        )

    def extract_keywords(self):
        path = self.cfg.working_dir_root + "/embeddings"
        self._init_model()
        extract_keywords(
            self.model, self.tokenizer, self.dataset, path, self.cfg.device
        )

    def analyze_keywords(self):
        path = self.cfg.working_dir_root + "/embeddings"

        analyze_keywords(path)

    def extract_st_doc_embeddings(self):
        path = self.cfg.root_path
        self.model = SentenceTransformer(
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )

        extract_st_doc_embeddings(self.model, self.dataset, path)

    def extract_e5_doc_embeddings(self):
        path = self.cfg.root_path
        self.model = SentenceTransformer(
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )

        extract_st_doc_embeddings(self.model, self.dataset, path)

    def setfit_train(self):
        setfit_train(self.dataset, self.cfg.label_scheme, "train")

    def setfit_predict(self):
        setfit_train(self.dataset, self.cfg.label_scheme, "predict")
