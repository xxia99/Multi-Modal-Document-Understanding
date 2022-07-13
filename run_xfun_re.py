#!/usr/bin/env python
# coding=utf-8

import logging
import os
import sys
from json import dump

import numpy as np
from datasets import ClassLabel, load_dataset
from PIL import Image, ImageDraw, ImageFont

import layoutlmft.data.datasets.xfun
import transformers
from layoutlmft import AutoModelForRelationExtraction
from layoutlmft.data.data_args import XFUNDataTrainingArguments
from layoutlmft.data.data_collator import DataCollatorForKeyValueExtraction
from layoutlmft.evaluation import re_score
from layoutlmft.models.model_args import ModelArguments
from layoutlmft.trainers import XfunReTrainer
from transformers import (
    AutoConfig,
    AutoTokenizer,
    HfArgumentParser,
    PreTrainedTokenizerFast,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint, is_main_process


logger = logging.getLogger(__name__)


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, XFUNDataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    logger.info(f"Training/evaluation parameters {training_args}")

    # Set seed before initializing model.
    set_seed(training_args.seed)
    datasets = load_dataset(
        os.path.abspath(layoutlmft.data.datasets.xfun.__file__),
        f"xfun.{data_args.lang}",
        additional_langs=data_args.additional_langs,
        keep_in_memory=True,
        download_mode="force_redownload",
    )

    if training_args.do_train:
        column_names = datasets["train"].column_names
        features = datasets["train"].features
    else:
        column_names = datasets["validation"].column_names
        features = datasets["validation"].features
    text_column_name = "input_ids"
    label_column_name = "labels"

    remove_columns = column_names

    # In the event the labels are not a `Sequence[ClassLabel]`, we will need to go through the dataset to get the
    # unique labels.
    def get_label_list(labels):
        unique_labels = set()
        for label in labels:
            unique_labels = unique_labels | set(label)
        label_list = list(unique_labels)
        label_list.sort()
        return label_list

    if isinstance(features[label_column_name].feature, ClassLabel):
        label_list = features[label_column_name].feature.names
        # No need to convert the labels since they are already ints.
        label_to_id = {i: i for i in range(len(label_list))}
    else:
        label_list = get_label_list(datasets["train"][label_column_name])
        label_to_id = {l: i for i, l in enumerate(label_list)}
    num_labels = len(label_list)

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=True,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    model = AutoModelForRelationExtraction.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    # Tokenizer check: this script requires a fast tokenizer.
    if not isinstance(tokenizer, PreTrainedTokenizerFast):
        raise ValueError(
            "This example script only works for models that have a fast tokenizer. Checkout the big table of models "
            "at https://huggingface.co/transformers/index.html#bigtable to find the model types that meet this "
            "requirement"
        )

    # Preprocessing the dataset
    # Padding strategy
    padding = "max_length" if data_args.pad_to_max_length else False

    if training_args.do_train:
        if "train" not in datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = datasets["train"]
        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(data_args.max_train_samples))

    if training_args.do_eval:
        if "validation" not in datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = datasets["validation"]
        example_ids = []
        for example_index in range(len(eval_dataset)):
            example = eval_dataset[example_index]
            example_ids.append(example['id'])
        if data_args.max_val_samples is not None:
            eval_dataset = eval_dataset.select(range(data_args.max_val_samples))

    if training_args.do_predict:
        if "test" not in datasets:
            raise ValueError("--do_predict requires a test dataset")
        test_dataset = datasets["validation"]
        if data_args.max_test_samples is not None:
            test_dataset = test_dataset.select(range(data_args.max_test_samples))

    # Data collator
    data_collator = DataCollatorForKeyValueExtraction(
        tokenizer,
        pad_to_multiple_of=8 if training_args.fp16 else None,
        padding=padding,
        max_length=512,
    )

    def compute_metrics(p):
        pred_relations, gt_relations = p
        score = re_score(pred_relations, gt_relations, mode="boundaries")
        return score

    # Initialize our Trainer
    trainer = XfunReTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # Training
    if training_args.do_train:
        checkpoint = last_checkpoint if last_checkpoint else None
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        trainer.save_model()  # Saves the tokenizer too for easy upload

        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        pred_relations, metrics = trainer.evaluate()

        max_val_samples = data_args.max_val_samples if data_args.max_val_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_val_samples, len(eval_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

        '''
        Inference Part
        '''
        predict_results = []
        gt_pairs = []
        pred_pairs = []
        entity_labels = eval_dataset.features['entities'].feature['label'].names
        labels = eval_dataset.features['labels'].feature.names
        label2color = {'question': 'blue', 'answer': 'green', 'header': 'orange', 'other': 'violet'}
        id2label = {v: k for v, k in enumerate(labels)}

        def iob_to_label(label):
            label = label[2:]
            if not label:
                return 'other'
            return label

        def unnormalize_box(bbox, width, height):
            return [
                width * (bbox[0] / 1000),
                height * (bbox[1] / 1000),
                width * (bbox[2] / 1000),
                height * (bbox[3] / 1000),
            ]

        for example_index in range(len(eval_dataset)):
            example = eval_dataset[example_index]
            # print(example.keys())
            predict_result = {}
            predict_result['document_id'] = example_ids[example_index]
            ids_list = example['input_ids']
            word_list = []  # 把所有字的id转换为字并存入word_list
            for index in ids_list:
                word_list.append(tokenizer._convert_id_to_token(index))
            words_num = len(example['entities']['end'])  # 60个词words，459个字word
            words_list = []  # 把所有语义实体从id转换为词，并存入words_list
            for i in range(words_num):
                start_ind = example['entities']['start'][i]
                end_ind = example['entities']['end'][i]
                words = ""
                for ind in range(start_ind, end_ind):
                    id = ids_list[ind]
                    if id != 6 and id != 52:
                        words += str(tokenizer._convert_id_to_token(id))
                words_list.append(words)
            label_ids = example['entities']['label']
            label_list = []  # 把每个语义实体的label从数字id转化为字符串
            for id in label_ids:
                label_list.append(entity_labels[id])

            lv_list = []  # 标注文档的label-words对
            for i in range(words_num):
                lv_list.append({label_list[i]: words_list[i]})

            head_ids = example['relations']['head']  # question在words_list里面的索引值
            tail_ids = example['relations']['tail']  # answer在words_list里面的索引值
            relation_num = len(head_ids)

            qa_pairs = {}  # 标注文档的question-answer对，即ground truth
            for i in range(relation_num):
                qstr = words_list[head_ids[i]]
                astr = words_list[tail_ids[i]]
                if qstr[0] == '▁':
                    qstr = qstr[1:]
                if qstr[-1] == ':':
                    qstr = qstr[:-1]
                if astr[0] == '▁':
                    astr = astr[1:]
                qa_pairs[qstr] = astr
            qa_pairs = dict(sorted(qa_pairs.items(), key=lambda x: x[0]))
            predict_result['Ground Truth'] = qa_pairs

            pred_qa_pairs = {}  # 预测出的question-answer对，即prediction
            relations = pred_relations[example_index]
            for relation in relations:
                qstr = words_list[relation['head_id']]
                astr = words_list[relation['tail_id']]
                if qstr[0] == '▁':
                    qstr = qstr[1:]
                if qstr[-1] == ':':
                    qstr = qstr[:-1]
                if astr[0] == '▁':
                    astr = astr[1:]
                pred_qa_pairs[qstr] = astr
            pred_qa_pairs = dict(sorted(pred_qa_pairs.items(), key=lambda x: x[0]))
            predict_result['Prediction'] = pred_qa_pairs
            predict_results.append(predict_result)

            image = Image.open("zh.val/" + predict_result['document_id'][:-2] + ".jpg")
            image = image.convert("RGB")
            draw = ImageDraw.Draw(image)
            font = ImageFont.load_default()
            width, height = image.size

            for word, box, label in zip(word_list, example['bbox'], example['labels']):
                actual_label = iob_to_label(id2label[label]).lower()
                box = unnormalize_box(box, width, height)
                draw.rectangle(box, outline=label2color[actual_label], width=2)
                draw.text((box[0] + 10, box[1] - 10), actual_label, fill=label2color[actual_label], font=font)
            image.save('processed/{name}'.format(name=predict_result['document_id'][:-2] + ".jpg"))

            # break
        with open('predict_results.txt', 'w') as f:
            dump(predict_results, f, ensure_ascii=False)


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
