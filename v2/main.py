import os
from pydoc import locate
import re
import random

import numpy as np

from sklearn.metrics import (
    classification_report,
    accuracy_score,
    f1_score,
    roc_auc_score,
    precision_score,
    recall_score,
    average_precision_score,
)

from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    BitsAndBytesConfig,
    EarlyStoppingCallback,
    AutoConfig,
)

import torch
from torch.nn import BCEWithLogitsLoss, Sigmoid, Linear
from torch import Tensor, cuda
from accelerate import infer_auto_device_map, init_empty_weights


from .model import GeminiModel
from .labels import get_label_scheme
from .data import get_dataset, get_gemini_data
from .dataloader import (
    custom_train_dataloader,
    custom_eval_dataloader,
    custom_test_dataloader,
)
from .training_args import CustomTrainingArguments
from .modes.extract_embeddings import extract_doc_embeddings
from .modes.extract_keywords import extract_doc_keywords
from .utils import log_gpu_memory, infer_device_map, preprocess_logits_for_metrics

current_optimal_threshold = 0.5  # Used in hierarchical loss (now obsolete)


def run(options):
    # Common variables

    # Make process deterministic
    torch.manual_seed(options.seed)
    np.random.seed(options.seed)
    random.seed(options.seed)
    torch.cuda.manual_seed_all(options.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    options.test = options.train if not options.test else options.test
    model_name = options.model_name
    working_dir = f"{options.output_path}/{options.train}_{options.test}{'_'+options.hp_search_lib if options.mode == 'hp_search' else ''}/{model_name.replace('/', '_')}"
    peft_modules = options.peft_modules.split(",") if options.peft_modules else None
    num_gpus = cuda.device_count()

    # Labels
    label_scheme = get_label_scheme(options.labels)

    print(f"Predicting {len(label_scheme)} labels.")

    # Torch dtype

    torch_dtype = (
        locate(f"torch.{options.torch_dtype}")
        if options.torch_dtype not in [None, "auto"]
        else options.torch_dtype
    )

    # Tf32

    if options.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # Imports based on options

    if options.use_flash_attention_2:
        from flash_attn import flash_attn_qkvpacked_func, flash_attn_func

    if options.peft:
        from peft import (
            LoraConfig,
            get_peft_model,
            TaskType,
            prepare_model_for_kbit_training,
            PeftModel,
            PeftConfig,
        )

    # Wandb setup

    try:
        from dotenv import load_dotenv

        load_dotenv()
        wandb_project_name = (
            f"{options.train}_{options.test}{'_tuning' if options.mode == 'hp_search' else ''}_{model_name.replace('/', '_')}"
            if not options.wandb_project
            else options.wandb_project
        )

        os.environ["WANDB_PROJECT"] = wandb_project_name
        os.environ["WANDB_API_KEY"] = os.getenv("WANDB_API_KEY", "")
        os.environ["WANDB_WATCH"] = "all"

        import wandb

        wandb.login()
        print("Using wandb")
    except:
        print("No wandb!")
        pass

    print(f"Imports finished!")

    # Tokenizer

    tokenizer_cnf = {
        "torch_dtype": torch_dtype,
        "cache_dir": f"{options.output_path}/tokenizer_cache",
    }

    tokenizer_cnf["low_cpu_mem_usage"] = not options.high_cpu_mem_usage

    if options.add_prefix_space:
        tokenizer_cnf["add_prefix_space"] = True

    if options.llm:
        tokenizer_cnf["add_eos_token"] = True
        tokenizer_cnf["add_bos_token"] = True

    if options.left_padding:
        tokenizer_cnf["padding_side"] = "left"

    if options.use_slow:
        tokenizer_cnf["use_fast"] = False
    if options.not_legacy:
        tokenizer_cnf["legacy"] = False
    if options.add_special_tokens:
        tokenizer_cnf["add_special_tokens"] = True

    tokenizer = (
        AutoTokenizer.from_pretrained(
            model_name if not options.custom_tokenizer else options.custom_tokenizer,
            **tokenizer_cnf,
        )
        if not options.gemini
        else None
    )

    # Some LLM's require a pad id
    if options.set_pad_id or options.llm:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.pad_token = tokenizer.eos_token

    # Dataset

    dataset = (
        get_dataset(
            options.train,
            options.test,
            options.labels,
            options.output_path,
            few_shot=options.few_shot,
        )
        if not options.gemini
        else get_gemini_data(options.train)
    )

    # If plotting, stop here

    if options.mode == "plot":
        from .modes import visualizations

        func = getattr(visualizations, options.plot)
        func(dataset)
        return

    # Shuffle and tokenize

    if options.mode == "evaluate":
        dataset.pop("train")

    if options.only_test:
        dataset.pop("dev")

    dataset = dataset.shuffle(seed=options.seed)
    dataset = (
        dataset.map(
            lambda example: tokenizer(
                example["text"],
                truncation=True,
                max_length=options.max_length,
                return_tensors=options.return_tensors,
            ),
            batched=True,
        )
        if not options.gemini
        else dataset
    )

    print("Data prepared!")

    # Override Trainer

    class MultilabelTrainer(Trainer):
        def __init__(self, *args, **kwargs):
            super(MultilabelTrainer, self).__init__(*args, **kwargs)

        if options.balance:

            def get_train_dataloader(self):
                return custom_train_dataloader(self)

            def get_eval_dataloader(self, eval_dataset=None):
                return custom_eval_dataloader(self, options.eval_batch_size)

            def get_test_dataloader(self, test_dataset=dataset["test"]):
                return custom_test_dataloader(
                    self, options.eval_batch_size, dataset["test"]
                )

        def compute_loss(self, model, inputs, return_outputs=False):
            print(self.optimizer)

            labels = inputs.pop("labels")
            outputs = model(**inputs)
            try:
                logits = outputs.logits
            except:
                logits = outputs.get("logits")

            if options.loss:
                loss_params = {
                    "gamma": self.args.loss_gamma,
                    "alpha": self.args.loss_alpha,
                }
                if options.loss == "HierarchicalBCEFocalLoss":
                    loss_params["threshold"] = current_optimal_threshold
                    loss_params["hierarchy_penalty_weight"] = options.loss_penalty

                loss_cls = locate(f"v2.loss.{options.loss}")
                loss_fct = loss_cls(**loss_params)
            else:
                loss_fct = BCEWithLogitsLoss()

            loss = loss_fct(
                logits.view(-1, self.model.config.num_labels),
                labels.float().view(-1, self.model.config.num_labels),
            )

            return (loss, outputs) if return_outputs else loss

    # Compute optimal threshold

    def optimize_threshold(predictions, labels):
        sigmoid = Sigmoid()
        probs = sigmoid(Tensor(predictions))
        best_f1 = 0
        best_f1_threshold = 0.5
        y_true = labels
        for th in np.arange(0.3, 0.7, 0.05):
            y_pred = np.zeros(probs.shape)
            y_pred[np.where(probs >= th)] = 1
            f1 = f1_score(y_true=y_true, y_pred=y_pred, average="micro")
            if f1 > best_f1:
                best_f1 = f1
                best_f1_threshold = th

        # This is used by the hierarchical loss, so we'll use global
        global current_optimal_threshold
        current_optimal_threshold = best_f1_threshold
        return best_f1_threshold

    # Calculate metrics

    def compute_metrics_fn(labels, predictions, return_preds=False):
        print("labels:")
        print(labels)
        print("preds:")
        print(predictions)
        threshold = (
            options.threshold
            if options.threshold
            else optimize_threshold(predictions, labels)
        )
        sigmoid = Sigmoid()
        probs = sigmoid(Tensor(predictions))
        y_pred = np.zeros(probs.shape)
        y_pred[np.where(probs >= threshold)] = 1
        y_th05 = np.zeros(probs.shape)
        y_th05[np.where(probs >= 0.5)] = 1
        try:
            roc_auc = roc_auc_score(labels, y_pred, average="micro")
        except:
            roc_auc = 0

        # Compute precision and recall
        precision = precision_score(labels, y_pred, average="micro")
        recall = recall_score(labels, y_pred, average="micro")

        # Compute PR AUC
        try:
            pr_auc = average_precision_score(labels, probs, average="micro")
        except:
            pr_auc = 0
        accuracy = accuracy_score(labels, y_pred)
        metrics = {
            "f1": f1_score(y_true=labels, y_pred=y_pred, average="micro"),
            "f1_th05": f1_score(y_true=labels, y_pred=y_th05, average="micro"),
            "precision": precision,
            "recall": recall,
            "pr_auc": pr_auc,
            "roc_auc": roc_auc,
            "accuracy": accuracy,
            "threshold": threshold,
        }
        # print(
        #    classification_report(labels, y_pred, target_names=label_scheme, digits=4)
        # )
        if not return_preds:
            return metrics
        else:
            return metrics, y_pred

    # Compute metrics (used by Trainer)

    def compute_metrics(p):
        log_gpu_memory()
        _, labels = p
        predictions = (
            p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        )

        return compute_metrics_fn(labels, predictions)

    # Print evaluation results

    def evaluate():
        if not options.only_test:
            print("Evaluating with dev set...")
            print(trainer.evaluate(dataset["dev"]))

        print("Evaluating with test set...")
        p = trainer.predict(dataset["test"])
        predictions = (
            p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        )
        metrics, preds = compute_metrics_fn(p.label_ids, predictions, return_preds=True)
        print(metrics)
        print(
            classification_report(
                p.label_ids, preds, target_names=label_scheme, digits=4
            )
        )

    # Initialize model

    def model_init():
        params = {
            "num_labels": len(label_scheme),
            "cache_dir": f"model_cache",
            "trust_remote_code": True,
            "offload_folder": "offload" if options.offload else None,
            "low_cpu_mem_usage": not options.high_cpu_mem_usage,
            "torch_dtype": torch_dtype,
            "device_map": options.device_map or None,
        }
        if options.infer_device_map:
            params["device_map"] = infer_device_map()
        if options.infer_auto_device_map:
            config = AutoConfig.from_pretrained(model_name)
            with init_empty_weights():
                model = locate(f"transformers.{options.transformer_model}").from_config(
                    config
                )

            params["device_map"] = infer_auto_device_map(model, dtype=torch_dtype)

        if options.use_flash_attention_2:
            params["attn_implementation"] = "flash_attention_2"
            params["use_flash_attention_2"] = True
        if options.quantize:
            params["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch_dtype,
            )
        if options.ignore_mismatched_sizes:
            params["ignore_mismatched_sizes"] = True

        model = (
            locate(f"transformers.{options.transformer_model}").from_pretrained(
                model_name, **params
            )
            if not options.gemini
            else GeminiModel()
        )

        if options.set_pad_id:
            model.config.pad_token_id = model.config.eos_token_id

        if options.peft:
            print("Using PEFT")

            # model.config.pretraining_tp = 1  # Set max linear layers (llama2)

            model_modules = str(model.modules)
            pattern = r"\((\w+)\): Linear"
            linear_layer_names = re.findall(pattern, model_modules)

            names = []
            # Print the names of the Linear layers
            for name in linear_layer_names:
                names.append(name)
            target_modules = list(set(names))

            print(f"Linear layers:\n {target_modules}")

            # Define LoRA Config
            lora_config = LoraConfig(
                r=options.lora_rank,
                lora_alpha=options.lora_alpha,
                target_modules=target_modules if not peft_modules else peft_modules,
                lora_dropout=options.lora_dropout,
                bias=options.lora_bias,
                task_type=TaskType.SEQ_CLS,
            )

            # add LoRA adaptor
            model.config.use_cache = not options.no_cache
            if options.gradient_checkpointing:
                model.gradient_checkpointing_enable()

            if options.kbit:
                model = prepare_model_for_kbit_training(model)
            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()

            if options.mode == "evaluate" and options.adapter_model_path:
                model = PeftModel.from_pretrained(
                    model, options.adapter_model_path  # , **params
                )
                # model = model.merge_and_unload()

        if options.add_classification_head:
            # Add a classification head on top of the model
            model.resize_token_embeddings(len(tokenizer))
            model.classifier = Linear(model.config.hidden_size, len(label_scheme))

        # Resize embedding size as we added bos token
        if options.llm:
            if model.config.vocab_size < len(tokenizer.get_vocab()):
                model.resize_token_embeddings(len(tokenizer.get_vocab()))

        print("Model initialized!")

        return model

    # Init trainer

    trainer = MultilabelTrainer(
        model=None if options.mode == "hp_search" else model_init(),
        model_init=model_init if options.mode == "hp_search" else None,
        args=CustomTrainingArguments(
            options.loss_gamma,
            options.loss_alpha,
            f"{working_dir}/checkpoints",
            overwrite_output_dir=True if not options.resume else False,
            evaluation_strategy=options.iter_strategy,
            save_strategy=options.iter_strategy,
            logging_strategy=options.iter_strategy,
            load_best_model_at_end=True,
            eval_steps=options.eval_steps,
            logging_steps=options.logging_steps,
            save_steps=options.save_steps,
            save_total_limit=options.save_total_limit,
            weight_decay=options.weight_decay,
            warmup_steps=options.warmup_steps,
            warmup_ratio=options.warmup_ratio,
            learning_rate=options.lr,
            max_grad_norm=options.max_grad_norm,
            lr_scheduler_type=options.lr_scheduler_type,
            metric_for_best_model=options.metric_for_best_model,
            greater_is_better=False
            if "loss" in options.metric_for_best_model
            else True,
            per_device_train_batch_size=int(options.train_batch_size / (num_gpus or 1)),
            per_device_eval_batch_size=int(options.eval_batch_size / (num_gpus or 1)),
            num_train_epochs=options.epochs,
            gradient_checkpointing=options.gradient_checkpointing,
            gradient_accumulation_steps=options.gradient_steps,
            optim=options.optim,
            fp16=options.fp16,
            bf16=options.bf16,
            tf32=options.tf32,
            group_by_length=True,
            resume_from_checkpoint=True if options.resume else None,
            eval_accumulation_steps=options.eval_accumulation_steps,
            auto_find_batch_size=options.auto_find_batch_size or False,
        ),
        train_dataset=dataset.get("train", []),
        eval_dataset=dataset.get("dev", []),
        compute_metrics=compute_metrics,
        data_collator=DataCollatorWithPadding(
            tokenizer=tokenizer, padding="longest", max_length=options.max_length
        )
        if not options.gemini
        else None,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=options.patience)],
        preprocess_logits_for_metrics=preprocess_logits_for_metrics
        if options.preprocess_logits
        else None,
    )

    print(f"Trainer prepared! Using {trainer.args._n_gpu} GPUs.")

    # Mode: just evaluate

    if options.mode == "evaluate":
        print("Evaluating pretrained model...")
        evaluate()

    # Mode: extract document embeddings

    elif options.mode == "extract_embeddings":
        print("Extracting document embeddings...")
        model = model_init()
        extract_doc_embeddings.extract(model, dataset, working_dir)

    # Mode: extract document keywords

    if options.mode == "extract_keywords":
        print("Extracting keywords...")
        model = model_init()
        dataset.set_format(type="torch")
        model = model.to("cpu")
        extract_doc_keywords(model, dataset, tokenizer, f"{working_dir}/keywords2.tsv")

    # Mode: train

    elif options.mode == "train":
        print("Training...")
        trainer.train()

        evaluate()

        if options.save_model:
            trainer.model.save_pretrained(f"{working_dir}/saved_model")

    # Mode: hyperparameter search

    elif options.mode == "hp_search":
        from .modes.hyperparameter_search import hyperparameter_search

        hyperparameter_search(
            trainer,
            options.hp_search_lib,
            working_dir,
            wandb_project_name,
            num_gpus,
            options.ray_log_path,
            options.min_lr,
            options.max_lr,
        )
