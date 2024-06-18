import csv
import glob
import json
import os
import random
import shutil
from pydoc import locate

import numpy as np
import torch
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from scipy.special import expit as sigmoid
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    f1_score,
    precision_recall_fscore_support,
)
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from .data import balanced_dataloader, get_dataset
from .labels import (
    decode_binary_labels,
    label_schemes,
    subcategory_to_parent_index,
    map_to_xgenre_binary,
    upper_all_indexes,
    upper_all_indexes_en,
    get_binary_representations,
)


def get_linear_modules(model):

    linear_modules = set()

    for name, module in model.named_modules():
        name = name.lower()
        if "attention" in name and "self" in name and "Linear" in str(type(module)):
            linear_modules.add(name.split(".")[-1])

    print(f"\nFound linear modules: {linear_modules}")
    return list(linear_modules)


def run(cfg):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # login to huggingface to get access to mixtral
    from huggingface_hub import login
    access_token_read = "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"
    login(token = access_token_read)

    # Make process deterministic
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    test_language = ""  # Used when predicting
    test_dataset = []  # Used when predicting
    multilabel_exclusion_stats = {"excluded": 0, "included": 0}
    pred_suffix = ("_"+cfg.test) if "multi" in cfg.test else ""

    # CUDA events for timing
    if device == "cuda":
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

    label_scheme = label_schemes[cfg.labels]
    prediction_label_scheme = label_schemes[cfg.predict_labels]
    print(f"Predicting {len(label_scheme)} labels")
    predict_upper_using_full = cfg.labels == "all" and cfg.predict_labels == "upper"
    predict_upper_en_using_full_en = (
        cfg.labels == "en_all" and cfg.predict_labels == "en_upper"
    )
    predict_xgenre_using_full = cfg.labels == "all" and cfg.predict_labels == "xgenre"
    model_output_dir = f"{cfg.model_output}/{cfg.model_name}{('_'+cfg.path_suffix) if cfg.path_suffix else ''}/labels_{cfg.labels}/{cfg.train}_{cfg.dev}/seed_{cfg.seed}{('/fold_'+str(cfg.use_fold)) if cfg.use_fold else ''}{('/subset_'+str(cfg.sample_subset)) if cfg.sample_subset else ''}"
    results_output_dir = f"{cfg.predictions_output}/{cfg.model_name}{('_'+cfg.path_suffix) if cfg.path_suffix else ''}/{cfg.train}_{cfg.dev}/seed_{cfg.seed}{('/fold_'+str(cfg.use_fold)) if cfg.use_fold else ''}{('/subset_'+str(cfg.sample_subset)) if cfg.sample_subset else ''}"
    print(
        f"This run {'saves models to' if not cfg.just_evaluate else 'uses model from'} {model_output_dir}"
    )
    print(f"Results are logged to {results_output_dir}")
    torch_dtype = locate(f"torch.{cfg.torch_dtype}")
    if not torch.cuda.is_available():
        torch_dtype = torch.float32
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    if "mixtral" in cfg.model_name.lower():
        tokenizer.pad_token = tokenizer.eos_token
    dataset = get_dataset(cfg, tokenizer)

    base_model_path = (
        model_output_dir if cfg.just_evaluate and not cfg.peft else cfg.model_name
    )

    nf4_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch_dtype,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        base_model_path,
        num_labels=len(label_scheme),
        torch_dtype=torch_dtype,
        use_flash_attention_2=cfg.fa2,
        quantization_config=nf4_config if cfg.nf4 else None,
        device_map="auto" if "mixtral" in cfg.model_name.lower() else None,
    )

    label2id = {label: id for id, label in enumerate(label_scheme)}
    id2label = {id: label for label, id in label2id.items()}

    model.config.label2id = label2id
    model.config.id2label = id2label

    if cfg.peft:
        if cfg.just_evaluate: 
            model = PeftModel.from_pretrained(model=model, model_id=model_output_dir) # , torch_device=device, offload_state_dict=False I added but did not help
            # eka on se huggingface malli ja sit se toinen se adapteri mikä tallennettuna
            print("peft model loaded")
        else:
            print("Using LoRa")
            model = get_peft_model(
                model,
                LoraConfig(
                    r=cfg.lora_rank,
                    lora_alpha=cfg.lora_alpha,
                    target_modules=(
                        get_linear_modules(model)
                        if not cfg.target_modules
                        else cfg.target_modules.split(",")
                    ),
                    lora_dropout=0.1,
                    bias="none",
                    task_type=TaskType.SEQ_CLS,
                ),
            )

    class CustomEarlyStoppingCallback(EarlyStoppingCallback):
        def __init__(
            self,
            early_stopping_patience: int = 1,
            early_stopping_threshold: float = 0.0,
        ):
            super().__init__(early_stopping_patience, early_stopping_threshold)
            self.best_epoch = 0

        def check_metric_value(self, args, state, control, metric_value):
            # best_metric is set by code for load_best_model
            operator = np.greater if args.greater_is_better else np.less
            if state.best_metric is None or (
                operator(metric_value, state.best_metric)
                and abs(metric_value - state.best_metric)
                > self.early_stopping_threshold
            ):
                self.early_stopping_patience_counter = 0
                self.best_epoch = state.global_step  # Update the best epoch
            else:
                self.early_stopping_patience_counter += 1

    class MultiLabelTrainer(Trainer):
        def __init__(self, *args, **kwargs):
            super(MultiLabelTrainer, self).__init__(*args, **kwargs)

        def compute_loss(self, model, inputs, return_outputs=False):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits
            BCE_loss = F.binary_cross_entropy_with_logits(
                logits, labels.float(), reduction="none"
            )
            pt = torch.exp(-BCE_loss)
            loss = cfg.loss_alpha * (1 - pt) ** cfg.loss_gamma * BCE_loss

            # Class balancing
            loss = loss * (
                labels * cfg.loss_alpha + (1 - labels) * (1 - cfg.loss_alpha)
            )
            loss = loss.mean()

            return (loss, outputs) if return_outputs else loss

        if (len(cfg.train.split("-")) > 1 or cfg.balanced_dataloader) and not cfg.just_evaluate:

            def get_train_dataloader(self):
                return balanced_dataloader(self, "train", cfg.train_batch_size)

        if (len(cfg.dev.split("-")) > 1 or cfg.balanced_dataloader) and not cfg.just_evaluate:

            def get_eval_dataloader(self, eval_dataset=None):
                return balanced_dataloader(self, "eval", cfg.eval_batch_size)

    def compute_metrics(p):
        true_labels = p.label_ids
        predictions = sigmoid(
            p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        )

        if cfg.labels in ["all", "all_mix"]:
            # Ensure that subcategory has corresponding parent category
            for i in range(predictions.shape[0]):
                for (
                    subcategory_index,
                    parent_index,
                ) in subcategory_to_parent_index.items():
                    if predictions[i, parent_index] < predictions[i, subcategory_index]:
                        predictions[i, parent_index] = predictions[i, subcategory_index]

        if predict_upper_using_full:
            true_labels = true_labels[:, upper_all_indexes]
            predictions = predictions[:, upper_all_indexes]
        elif predict_upper_en_using_full_en:
            true_labels = true_labels[:, upper_all_indexes_en]
            predictions = predictions[:, upper_all_indexes_en]

        best_threshold, best_f1 = 0, 0
        for threshold in np.arange(0.3, 0.7, 0.05):
            binary_predictions = predictions > threshold
            f1 = f1_score(true_labels, binary_predictions, average="micro")
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold

        binary_predictions = predictions > best_threshold

        if predict_xgenre_using_full:
            true_labels, binary_predictions = map_to_xgenre_binary(
                true_labels, binary_predictions
            )

        if cfg.multilabel_eval:
            
            """
            True    Pred
            ======  ======
            Any     Single   
            Any     Hybrid   
            Single  Any      
            Single  Single   
            Single  Hybrid   
            Hybrid  Any      
            Hybrid  Single   
            Hybrid  Hybrid   
                        
            """

            # Get row indices for binary representations of multilabel predictions
            non_hybrids = get_binary_representations(cfg.predict_labels)
            exclude_indexes = []

            # Get the choices as a list
            true_filter, pred_filter = cfg.multilabel_eval.split("_")
        
            for i, example in enumerate(true_labels):
                if true_filter == "single":
                    if [int(val) for val in example] not in non_hybrids:
                        exclude_indexes.append(i)
                elif true_filter == "hybrid":
                    if [int(val) for val in example] in non_hybrids:
                        exclude_indexes.append(i)

            for i, example in enumerate(binary_predictions):
                if pred_filter == "single":
                    if [int(val) for val in example] not in non_hybrids:
                        exclude_indexes.append(i)
                elif pred_filter == "hybrid":
                    if [int(val) for val in example] in non_hybrids:
                        exclude_indexes.append(i)

            # Create a mask where only indices not in the list are True
            mask = np.ones(len(true_labels), dtype=bool)
            mask[exclude_indexes] = False

            # Filter predictions and true_labels
            binary_predictions = binary_predictions[mask]
            true_labels = true_labels[mask]

            multilabel_exclusion_stats["included"] += np.sum(mask)
            multilabel_exclusion_stats["excluded"] += np.sum(~mask)
        

        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels, binary_predictions, average="micro"
        )
        accuracy = accuracy_score(true_labels, binary_predictions)
        # pr_auc = average_precision_score(true_labels, predictions, average="micro")

        metrics = {
            "f1": f1,
            "f1_macro": f1_score(
                true_labels, binary_predictions, average="macro", zero_division=np.nan
            ),
            "precision": precision,
            "recall": recall,
            "accuracy": accuracy,
            # "pr_auc": pr_auc,
            "threshold": best_threshold,
        }

        if cfg.just_evaluate:

            cl_report_dict = classification_report(
                true_labels,
                binary_predictions,
                target_names=label_schemes[cfg.predict_labels],
                digits=4,
                output_dict=True,
            )
            metrics["label_scores"] = {
                key: val
                for key, val in cl_report_dict.items()
                if key in prediction_label_scheme
            }

            true_labels_str = decode_binary_labels(true_labels, cfg.labels)
            predicted_labels_str = decode_binary_labels(binary_predictions, cfg.labels)
            example_indices = [x["row"] for x in test_dataset]
            data = list(zip(true_labels_str, predicted_labels_str, example_indices))
            trues_and_probs = list(
                zip(true_labels, np.round(predictions, 4), example_indices)
            )
            if cfg.save_predictions:
                os.makedirs(results_output_dir, exist_ok=True)

                with open(
                    f"{results_output_dir}/{cfg.labels}_{cfg.predict_labels}_{test_language}{pred_suffix}{('_'+cfg.multilabel_eval) if cfg.multilabel_eval else ''}.tsv",
                    "w",
                    newline="",
                ) as csvfile:
                    csv_writer = csv.writer(csvfile, delimiter="\t")
                    csv_writer.writerows(data)

                with open(
                    f"{results_output_dir}/{cfg.labels}_{cfg.predict_labels}_{test_language}{pred_suffix}_probs{('_'+cfg.multilabel_eval) if cfg.multilabel_eval else ''}.tsv",
                    "w",
                    newline="",
                ) as csvfile:
                    csv_writer = csv.writer(csvfile, delimiter="\t")
                    csv_writer.writerows(trues_and_probs)

                with open(
                    f"{results_output_dir}/{cfg.labels}_{cfg.predict_labels}_{test_language}{pred_suffix}{('_'+cfg.multilabel_eval) if cfg.multilabel_eval else ''}_metrics.json",
                    "w",
                ) as f:
                    json.dump(metrics, f)

            print(metrics)

        return metrics

    early_stopping_callback = CustomEarlyStoppingCallback(
        early_stopping_patience=cfg.patience
    )

    trainer = MultiLabelTrainer(
        model=model,
        args=TrainingArguments(
            output_dir=model_output_dir,
            overwrite_output_dir=True,
            num_train_epochs=30,
            per_device_train_batch_size=cfg.train_batch_size,
            per_device_eval_batch_size=cfg.eval_batch_size,
            warmup_ratio=0.05,
            weight_decay=0.01,
            learning_rate=cfg.learning_rate,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            logging_strategy="epoch",
            gradient_accumulation_steps=cfg.grad_acc_steps,
            eval_accumulation_steps=8,
            metric_for_best_model="eval_loss",
            load_best_model_at_end=True,
            save_total_limit=2,
            tf32=True if torch.cuda.is_available() else False,
            group_by_length=True,
            report_to=None,
        ),
        train_dataset=dataset.get("train", []),
        eval_dataset=dataset.get("dev", []),
        compute_metrics=compute_metrics,
        callbacks=[early_stopping_callback],
    )

    if not cfg.just_evaluate:
        trainer.train()
        if not cfg.no_save:
            trainer.save_model()
        for dir_path in glob.glob(f"{model_output_dir}/checkpoint*"):
            shutil.rmtree(dir_path, ignore_errors=True)
        shutil.rmtree(f"{model_output_dir}/runs", ignore_errors=True)

        # Gather training parameters and metadata
        training_metadata = {
            "batch_size": trainer.args.per_device_train_batch_size,
            "learning_rate": trainer.args.learning_rate,
            "warmup_ratio": trainer.args.warmup_ratio,
            "stopped_epoch": early_stopping_callback.best_epoch,
            "total_epochs": trainer.state.epoch,
        }

        with open(f"{model_output_dir}/training_metadata.json", "w") as f:
            json.dump(training_metadata, f, indent=4)

    print("Predicting..")
    cfg.just_evaluate = True

    # get these initiated so I can get the mean
    latencies = []
    throughputs = []

    # add torch.compile() because it makes a difference in inference speeds! https://huggingface.co/docs/transformers/perf_torch_compile#a100-batch-size-1
    #model = torch.compile(model)

    test_languages = cfg.test.split("-") if "multi" not in cfg.test else list(set(dataset['test']['language']))
    for language in test_languages:

        print(f"-- {language} --")
        test_language = language
        test_dataset = dataset["test"].filter(
            lambda example: example["language"] == language,
            num_proc=None
        )
        print('filtered language')

        if cfg.sample:
            test_dataset = test_dataset.select(range(cfg.sample))

        if device == "cuda":

            print("inside cuda")

            # initiate list to keep track of batch sizes
            latency2 = []
            throughput2 = []

            # tokenize texts
            #text_inputs = []
            #for i in range(len(test_dataset)):
                #text_inputs.append(test_dataset[i]["text"])

            #inputs = tokenizer(text_inputs, return_tensors="pt", padding=True, truncation=True, max_length=512)
            #print("tokenized")

            device2 = torch.device("cuda:0")

             # do gpu warm up!!
            inputs2 = []
            for i in range(0,len(test_dataset), 1):
                inp = {'input_ids': torch.tensor([test_dataset[i]['input_ids']], device=device2), 'attention_mask': torch.tensor([test_dataset[i]['attention_mask']], device=device2)}
                inputs2.append(inp)

            print("changed inputs")

            if test_language == "en":
                with torch.no_grad():
                    print("gpu warm up")
                    for i in range(5):
                        for inp in inputs2:
                            predictions = model(**inp)

            for batch in range(1, 9):

                inputs2 = []
                print("batch", batch)
                for i in range(0,len(test_dataset), batch):
                    # correct shape
                    if batch == 1:
                        inp = {'input_ids': torch.tensor([test_dataset[i]['input_ids']], device=device2), 'attention_mask': torch.tensor([test_dataset[i]['attention_mask']], device=device2)}
                       # inp = {'input_ids': torch.tensor([inputs['input_ids'][i].tolist()], device=device2), 'attention_mask': torch.tensor([inputs['attention_mask'][i].tolist()], device=device2)}
                    elif batch > 1:
                        #inp = {'input_ids': torch.tensor(inputs['input_ids'][i:i+batch].tolist(), device=device2), 'attention_mask': torch.tensor(inputs['attention_mask'][i:i+batch].tolist(), device=device2)}
                        inp = {'input_ids': torch.tensor(test_dataset[i:i+batch]['input_ids'], device=device2), 'attention_mask': torch.tensor(test_dataset[i:i+batch]['attention_mask'], device=device2)}
                        # I think that is how it should work? if batch bigger than one take out the []surrounding the tensor
                    else:
                        raise ValueError('Batch should be one or bigger!')
                    inputs2.append(inp)

                #print(inputs2[0])
                print("number of examples",len(test_dataset))
                print("number of batches", len(inputs2))

                # repeat experiment and get mean value
                repetitions = 10 # 50 or 100
                timings=np.zeros((repetitions,1)) 
                total_time = 0

                with torch.no_grad():
                    print("predicting")
                    print("repetitions",repetitions)
                    for rep in range(repetitions):
                        start_event.record()
                        # TODO here I could add a loading bar?
                        for inp in inputs2: # the inputs2 is batched already
                            predictions = model(**inp) 

                        end_event.record()
                        torch.cuda.synchronize()
                        curr_time = start_event.elapsed_time(end_event)
                        total_time += curr_time
                        timings[rep] = curr_time

                total_samples = len(test_dataset) # or inputs or text_inputs
                
                mean_syn = np.sum(timings) / repetitions
                std_syn = np.std(timings)
                print("total time:", total_time)
                print("mean time for all repetitions:", mean_syn)
                print("mean standard deviation between all repetitions", std_syn)

                latency = total_time / (total_samples * repetitions) # elapsed_time / total_samples
                # Latency per sample in milliseconds

                throughput = (total_samples * repetitions) / ( # total_samples / elapsed_time
                    total_time / 1000 #should I divide by latency instead of elapsed time?
                ) # Throughput in samples per second

                print(f"mean latency per sample: {latency} ms")
                print(f"mean throughput: {throughput} samples/sec")
    
                latency2.append(latency)
                throughput2.append(throughput)

        else:
            trainer.predict(test_dataset)

        if cfg.multilabel_eval:
            print(
                f"Excluded {multilabel_exclusion_stats['excluded']} examples and kept {multilabel_exclusion_stats['included']} examples"
            )
        
        latencies.append(latency2)
        throughputs.append(throughput2)

    # print mean of latency and throughput
    print("----------------------------")
    for i in range(0,8):
        latency3 = [item[i] for item in latencies]
        throughput3 = [item[i] for item in throughputs]
        print("batch was", i+1)
        print("mean latency",np.mean(np.asarray(latency3)))
        print("mean throughput", np.mean(np.asarray(throughput3)))
