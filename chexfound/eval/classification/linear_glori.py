# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import toolz
import argparse
from functools import partial
import json
import logging
import os
import sys
from typing import List, Optional
import math
from glob import glob

import numpy as np
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist
from fvcore.common.checkpoint import Checkpointer, PeriodicCheckpointer
import pandas as pd

from chexfound.data import SamplerType, make_data_loader, make_dataset
from chexfound.data.transforms import make_classification_eval_transform, make_classification_train_transform
import chexfound.distributed as distributed
from chexfound.eval.metrics import MetricType, build_metric
from chexfound.eval.setup import get_args_parser as get_setup_args_parser
from chexfound.eval.setup import setup_and_build_model
from chexfound.eval.utils import (ModelWithIntermediateLayers, evaluate, apply_method_to_nested_values,
                                  make_datasets, make_data_loaders, extract_hyperparameters_from_model,
                                  is_padded_matrix, collate_fn_3d, str2bool, trainable_parameters, bitfit,
                                  evaluate_preds)
from chexfound.eval.classification.utils import (setup_linear_classifiers, setup_glori, LinearPostprocessor)
from chexfound.logging import MetricLogger
from chexfound.data.wrappers import FewShotDatasetWrapper, SystemicSamplerWrapper
from chexfound.models.vision_transformer import DinoVisionTransformer

from peft import LoraConfig, get_peft_model
from chexfound.eval.losses import AsymmetricLoss

from chexfound.data.datasets import CXRLT

logger = logging.getLogger("chexfound")
import torch
import torch.nn as nn
def debug_optimizer_param_shapes(feature_model, linear_classifiers, optimizer):
    """
    检查 feature_model 和 linear_classifiers 的参数与 optimizer 是否匹配
    """
    print("=== Debug Optimizer vs Model ===")
    model_params = list(feature_model.parameters()) + list(linear_classifiers.parameters())
    opt_params = []
    for g in optimizer.param_groups:
        opt_params.extend(g['params'])

    print(f"Total model params: {len(model_params)}")
    print(f"Total optimizer params: {len(opt_params)}")

    if len(model_params) != len(opt_params):
        print("⚠️ 参数数量不匹配！可能导致 _multi_tensor_adam 错误")
    else:
        print("参数数量匹配 ✅")

    # 检查每个参数形状
    for i, (mp, op) in enumerate(zip(model_params, opt_params)):
        if mp.shape != op.shape:
            print(f"⚠️ 参数 {i} 形状不匹配!")
            print(f"   model param shape: {mp.shape}")
            print(f"   optimizer param shape: {op.shape}")

def check_optimizer_model_match(model, optimizer):
    """
    检查 optimizer 和 model 参数是否匹配
    """
    model_params = list(model.parameters())
    opt_params = []
    for group in optimizer.param_groups:
        opt_params.extend(group['params'])

    print(f"模型参数数量: {len(model_params)}")
    print(f"优化器参数数量: {len(opt_params)}")

    if len(model_params) != len(opt_params):
        print("⚠️ 参数数量不匹配！")
    else:
        print("参数数量匹配 ✅")

    # 逐个检查形状
    for i, (p_model, p_opt) in enumerate(zip(model_params, opt_params)):
        if p_model.shape != p_opt.shape:
            print(f"⚠️ 参数 {i} 形状不匹配:")
            print(f"    model param shape: {p_model.shape}")
            print(f"    optimizer param shape: {p_opt.shape}")
        else:
            print(f"参数 {i} 形状匹配: {p_model.shape}")



def get_args_parser(
        description: Optional[str] = None,
        parents: Optional[List[argparse.ArgumentParser]] = [],
        add_help: bool = True,
):
    setup_args_parser = get_setup_args_parser(parents=parents, add_help=False)
    parents = [setup_args_parser]
    parser = argparse.ArgumentParser(
        description=description,
        parents=parents,
        add_help=add_help,
    )
    parser.add_argument(
        "--train-dataset",
        dest="train_dataset_str",
        type=str,
        help="Training dataset",
    )
    parser.add_argument(
        "--val-dataset",
        dest="val_dataset_str",
        type=str,
        help="Validation dataset",
    )
    parser.add_argument(
        "--test-dataset",
        dest="test_dataset_str",
        type=str,
        help="Test dataset",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--val-epochs",
        type=int,
        help="Number of epochs for testing on validation",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Batch Size (per GPU)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        help="Number de Workers",
    )
    parser.add_argument(
        "--epoch-length",
        type=int,
        help="Length of an epoch in number of iterations",
    )
    parser.add_argument(
        "--save-checkpoint-frequency",
        type=int,
        help="Number of epochs between two named checkpoint saves.",
    )
    parser.add_argument(
        "--eval-period-epochs",
        type=int,
        help="Number of epochs between two evaluations.",
    )
    parser.add_argument(
        "--learning-rates",
        nargs="+",
        type=float,
        help="Learning rates to grid search.",
    )
    parser.add_argument(
        "--backbone-learning-rate",
        type=float,
    )
    parser.add_argument(
        "--n-last-blocks",
        nargs="+",
        type=int
    )
    parser.add_argument(
        "--avgpools",
        nargs="+",
        type=str2bool
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Whether to not resume- from existing checkpoints",
    )
    parser.add_argument(
        "--val-metric-type",
        type=MetricType,
        choices=list(MetricType),
        help="Validation metric",
    )
    parser.add_argument(
        "--classifier-fpath",
        type=str,
        help="Path to a file containing pretrained linear classifiers",
    )
    parser.add_argument(
        "--fine-tune",
        type=bool,
        help="Whether to finetune the backbone.",
    )
    parser.add_argument(
        "--shots",
        nargs="+",
        type=int,
        help="Number of shots for each class.",
    )
    parser.add_argument(
        "--backbone",
        type=str,
        help="The name of the backbone model to use [chexfound, vit-large-imagenet21k]",
    )
    parser.add_argument(
        "--peft",
        type=str,
        help="The name of the peft technique to use [lora]",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        help="The size of input image"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        help="Num of samples to take from the dataset"
    )
    parser.add_argument(
        "--infer",
        action="store_true",
        help="Whether to perform inference from existing checkpoints",
    )
    parser.add_argument(
        "--focal-loss",
        action="store_true",
        help="Whether to use focal loss"
    )
    parser.add_argument(
        "--multiview",
        action="store_true",
        help="Whether to activate multiview"
    )  # only for linear probe
    parser.add_argument(
        "--glori",
        action="store_true",
        help="Whether to use multilabel decoder"
    )
    parser.add_argument(
        "--flip",
        action="store_true",
        help="Whether to use test time flip"
    )
    parser.add_argument(
        "--rot90-train",
        action="store_true",
        help="Whether to use rot90 at training"
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        choices=[1e-1, 1e-2, 1e-4],
        help="L2 regularization weight"
    )
    parser.add_argument(
        "--decoder-dim",
        type=int,
        choices=[384, 768],
        help="Multilabel decoder dimension"
    )
    parser.add_argument(
        "--cat-cls",
        action='store_true',
        help="whether to concatenate cls token"
    )
    parser.set_defaults(
        train_dataset_str="NIHChestXray:split=TRAIN",
        val_dataset_str=None,
        test_dataset_str="NIHChestXray:split=TEST",
        epochs=10,
        val_epochs=None,
        batch_size=128,
        num_workers=8,
        epoch_length=None,
        save_checkpoint_frequency=2,
        eval_period_epochs=2,
        learning_rates=[1e-5,  1e-4],
        backbone_learning_rate=1e-5,
        n_last_blocks=[4],
        avgpools=[False],
        val_metric_type=MetricType.MULTILABEL_AUROC,
        classifier_fpath=None,
        fine_tune=False,
        shots=None,
        backbone="chexfound",
        peft=None,
        image_size=224,
        num_samples=None,
        weight_decay=1e-4,
        decoder_dim=768,
    )
    return parser


def has_ddp_wrapper(m: nn.Module) -> bool:
    return isinstance(m, DistributedDataParallel)


def remove_ddp_wrapper(m: nn.Module) -> nn.Module:
    return m.module if has_ddp_wrapper(m) else m


def scale_lr(learning_rates, batch_size):
    return learning_rates * (batch_size * distributed.get_global_size()) / 256.0


@torch.no_grad()
def evaluate_linear_classifiers(
        feature_model,
        linear_classifiers,
        data_loader,
        metric_type,
        metrics_file_path,
        num_of_classes,
        iteration,
        prefixstring="",
        best_classifier_on_val=None,
        multiview=False,
):
    logger.info("running validation !")

    labels = list(data_loader.dataset.class_names)
    metric = build_metric(metric_type, num_classes=num_of_classes, labels=labels)
    postprocessors = {k: LinearPostprocessor(v) for k, v in linear_classifiers.classifiers_dict.items()}
    metrics = {k: metric.clone() for k in linear_classifiers.classifiers_dict}

    _, results_dict_temp = evaluate(
        feature_model,
        data_loader,
        postprocessors,
        metrics,
        torch.cuda.current_device(),
        multiview=multiview,
    )

    logger.info("")
    results_dict = {}
    max_score = 0
    best_classifier = ""
    eval_metric = str(list(metric)[0])

    for i, (classifier_string, metric) in enumerate(results_dict_temp.items()):
        logger.info(f"{prefixstring} -- Classifier: {classifier_string} * {metric}")
        if (
                best_classifier_on_val is None and metric[eval_metric].item() > max_score
        ) or classifier_string == best_classifier_on_val:
            max_score = metric[eval_metric].item()
            best_classifier = classifier_string

    results_dict["best_classifier"] = {"name": best_classifier, "results": apply_method_to_nested_values(
        results_dict_temp[best_classifier],
        method_name="item",
        nested_types=(dict))}

    logger.info(f"best classifier: {results_dict['best_classifier']}")

    if distributed.is_main_process():
        with open(metrics_file_path, "a") as f:
            f.write(f"{prefixstring}\n")
            for k, v in results_dict.items():
                f.write(json.dumps({k: v}) + "\n")
            f.write("\n")

    return results_dict


@torch.no_grad()
def evaluate_linear_classifiers_preds(
        feature_model,
        linear_classifiers,
        data_loader,
        metric_type,
        metrics_file_path,
        num_of_classes,
        iteration,
        prefixstring="",
        best_classifier_on_val=None,
        multiview=False,
):
    logger.info("running validation !")

    labels = list(data_loader.dataset.class_names)
    metric = build_metric(metric_type, num_classes=num_of_classes, labels=labels)
    postprocessors = {k: LinearPostprocessor(v) for k, v in linear_classifiers.classifiers_dict.items()}
    metrics = {k: metric.clone() for k in linear_classifiers.classifiers_dict}

    _, results_dict_temp, metric_inputs = evaluate_preds(
        feature_model,
        data_loader,
        postprocessors,
        metrics,
        torch.cuda.current_device(),
        multiview=multiview,
    )

    # for i, (classifier_string, metric) in enumerate(results_dict_temp.items()):
    #     # tensor to float
    #     metric_new = {}
    #     for k, v in metric["class_auroc"].items():
    #         metric_new[k] = float(v)
    #     logger.info(f"{prefixstring} -- Classifier: {classifier_string} * {metric_new}")

    for i, (classifier_string, metric) in enumerate(results_dict_temp.items()):
        logger.info(f"{prefixstring} -- Classifier: {classifier_string} * {metric}")

    return metric_inputs


def eval_linear(
        *,
        feature_model,
        linear_classifiers,
        train_data_loader,
        val_data_loader,
        metrics_file_path,
        optimizer,
        scheduler,
        output_dir,
        max_iter,
        checkpoint_period,  # In number of epochs, creates a new file every period
        running_checkpoint_period,  # Period to update main checkpoint file
        eval_period,
        metric_type,
        num_of_classes,
        resume=True,
        classifier_fpath=None,
        is_multilabel=True,
        focal_loss=False,
        multiview=False,
        peft=None,
):
    if feature_model.fine_tune or peft is not None:  # todo: for peft, need also save the feature model
        checkpointer = Checkpointer(nn.Sequential(feature_model, linear_classifiers), output_dir, optimizer=optimizer,
                                    scheduler=scheduler)
    else:
        checkpointer = Checkpointer(linear_classifiers, output_dir, optimizer=optimizer, scheduler=scheduler)
    start_iter = checkpointer.resume_or_load(classifier_fpath or "", resume=resume).get("iteration", -1) + 1
    if not (resume and checkpointer.has_checkpoint()):
        start_iter = 0

    periodic_checkpointer = PeriodicCheckpointer(checkpointer, checkpoint_period, max_iter=max_iter)
    iteration = start_iter
    logger.info("Starting training from iteration {}".format(start_iter))
    metric_logger = MetricLogger(delimiter="  ")
    header = "Training"

    for data, labels in metric_logger.log_every(
            train_data_loader,
            10,
            header,
            max_iter,
            start_iter,
    ):
        # data = data.cuda(non_blocking=True)
        labels = torch.tensor(labels).cuda(non_blocking=True)

        # forward pass
        if multiview:
            features = [feature_model.forward_(data[0].cuda(non_blocking=True)),
                        feature_model.forward_(data[1].cuda(non_blocking=True))]
        else:
            features = feature_model(data.cuda(non_blocking=True))
        outputs = linear_classifiers(features)

        # calculate loss
        if is_multilabel:
            losses = {}
            for k, v in outputs.items():
                weight = torch.where(labels < 0, 0, 1)
                labels = torch.where(labels < 0, 0, labels)
                losses[f"loss_{k}"] = nn.BCEWithLogitsLoss(weight=weight, reduction='sum')(v.float(), labels.float())
                if focal_loss:
                    losses[f"loss_{k}"] += AsymmetricLoss(weight=weight)(v.float(), labels.float())
        else:
            # loss_fn = nn.BCEWithLogitsLoss() if num_of_classes == 1 else nn.CrossEntropyLoss()
            loss_fn = nn.BCEWithLogitsLoss(reduction='sum') if num_of_classes == 1 else nn.CrossEntropyLoss(
                reduction='sum')
            losses = {f"loss_{k}": loss_fn(v, labels.long()) for k, v in outputs.items()}

        loss = sum(losses.values())

        # compute the gradients
        optimizer.zero_grad()
        loss.backward()

        # zero out nan grad
        for model in [feature_model, linear_classifiers]:
            for param in model.parameters():
                if param.grad is not None and torch.isnan(param.grad).any():
                    param.grad.zero_()

        # step
        optimizer.step()
        scheduler.step()

        # log
        if iteration % 10 == 0:
            torch.cuda.synchronize()
            metric_logger.update(loss=loss.item())
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])
            print("lr", optimizer.param_groups[0]["lr"])

        if iteration - start_iter > 5:
            if iteration % running_checkpoint_period == 0:
                torch.cuda.synchronize()
                if distributed.is_main_process():
                    logger.info("Checkpointing running_checkpoint")
                    periodic_checkpointer.save("running_checkpoint_linear_eval", iteration=iteration)
                torch.cuda.synchronize()
        periodic_checkpointer.step(iteration)

        if eval_period > 0 and (iteration + 1) % eval_period == 0 and iteration != max_iter - 1:
            _ = evaluate_linear_classifiers(
                feature_model=feature_model,
                linear_classifiers=remove_ddp_wrapper(linear_classifiers),
                data_loader=val_data_loader,
                metrics_file_path=metrics_file_path,
                prefixstring=f"ITER: {iteration} {val_data_loader.dataset.split.value}",
                metric_type=metric_type,
                num_of_classes=num_of_classes,
                iteration=iteration,
                multiview=multiview,
            )
            torch.cuda.synchronize()

        iteration = iteration + 1

    val_results_dict = evaluate_linear_classifiers(
        feature_model=feature_model,
        linear_classifiers=remove_ddp_wrapper(linear_classifiers),
        data_loader=val_data_loader,
        metrics_file_path=metrics_file_path,
        prefixstring=f"ITER: {iteration} {val_data_loader.dataset.split.value}",
        metric_type=metric_type,
        num_of_classes=num_of_classes,
        iteration=iteration,
        multiview=multiview,
    )
    return val_results_dict, feature_model, linear_classifiers, iteration


def run_eval_linear(
        model,
        output_dir,
        train_dataset_str,
        test_dataset_str,
        batch_size,
        epochs,
        val_epochs,
        epoch_length,
        num_workers,
        save_checkpoint_frequency,
        eval_period_epochs,
        learning_rates,
        backbone_learning_rate,
        n_last_blocks_list,
        avgpools,
        autocast_dtype,
        val_dataset_str=None,
        resume=True,
        classifier_fpath=None,
        val_metric_type=MetricType.MULTILABEL_AUROC,
        fine_tune=False,
        shots=None,
        backbone="chexfound",
        peft=None,
        image_size=224,
        num_samples=None,
        focal_loss=False,
        multiview=False,
        glori=False,
        weight_decay=1e-2,
        rot90_train=False,
        decoder_dim=768,
        cat_cls=False,
):
    seed = 0
    torch.manual_seed(seed)

    if test_dataset_str == None:
        raise ValueError("Test dataset cannot be None")

    if "resnet" in backbone or "vgg" in backbone or "dense" in backbone:
        n_last_blocks_list = [1]
        avgpools = [False]

    train_transform = make_classification_train_transform(crop_size=image_size, rot90=rot90_train)
    eval_transform = make_classification_eval_transform(resize_size=image_size, crop_size=image_size)
    train_dataset, val_dataset, test_dataset = make_datasets(train_dataset_str=train_dataset_str,
                                                             val_dataset_str=val_dataset_str,
                                                             test_dataset_str=test_dataset_str,
                                                             train_transform=train_transform,
                                                             eval_transform=eval_transform)
    if shots != None:
        logger.info(f"Running dataset in {shots}-shot setting")
        train_dataset = FewShotDatasetWrapper(train_dataset, shots=shots)
    elif num_samples != None:
        logger.info(f"Running dataset with {num_samples} samples only")
        train_dataset = SystemicSamplerWrapper(train_dataset, num_samples=num_samples)

    batch_size = train_dataset.__len__() if batch_size > train_dataset.__len__() else batch_size
    num_of_classes = 2
    logger.info(f"{num_of_classes} classes")
    # num_of_classes = 1 if num_of_classes == 2 else num_of_classes
    is_multilabel = test_dataset.is_multilabel()
    is_3d = test_dataset.is_3d()
    collate_fn = None if not is_3d else collate_fn_3d

    n_last_blocks = max(n_last_blocks_list)
    autocast_ctx = partial(torch.cuda.amp.autocast, enabled=True, dtype=autocast_dtype)
    feature_model = ModelWithIntermediateLayers(model, n_last_blocks, autocast_ctx, is_3d=is_3d, fine_tune=fine_tune)

    sample_input = train_dataset[0][0][0] if is_3d else train_dataset[0][0]
    if multiview:
        sample_input = train_dataset[0][0][0][0] if is_3d else train_dataset[0][0][0]
    sample_input = sample_input.unsqueeze(0).cuda()
    sample_output = feature_model.forward_(sample_input)

    if epoch_length == None:
        epoch_length = math.ceil(train_dataset.__len__() / (batch_size * dist.get_world_size()))
    eval_period_iterations = eval_period_epochs * epoch_length
    checkpoint_period = save_checkpoint_frequency * epoch_length

    if not glori:
        linear_classifiers, optim_param_groups = setup_linear_classifiers(
            sample_output=sample_output,
            n_last_blocks_list=n_last_blocks_list,
            learning_rates=learning_rates,
            avgpools=avgpools,
            num_classes=num_of_classes,
            is_3d=is_3d,
            multiview=multiview,
        )
    else:
        linear_classifiers, optim_param_groups = setup_glori(
            sample_output=sample_output,
            n_last_blocks_list=n_last_blocks_list,
            learning_rates=learning_rates,
            avgpools=avgpools,
            num_classes=num_of_classes,
            multiview=multiview,
            decoder_dim=decoder_dim,
            cat_cls=cat_cls,
        )

    if val_epochs is not None:
        max_iter = epoch_length * val_epochs
    else:
        max_iter = epoch_length * epochs

    if fine_tune:
        logger.info("Finetuning backbone")
        optim_param_groups.append({'params': feature_model.parameters(), 'lr': backbone_learning_rate})
        checkpoint_model = nn.Sequential(feature_model, linear_classifiers)
    elif peft == "lora":
        logger.info("Using LoRA for fine tuning")
        config = LoraConfig(
            r=16,
            lora_alpha=16,
            # target_modules=["qkv", "fc1", "fc2"],  # todo: for peft, this has to look closely
            target_modules=["qkv", 'proj'],
            lora_dropout=0.1,
            bias="lora_only",
            modules_to_save=["classifier"],
        )
        feature_model = get_peft_model(feature_model, config)
        tp, ap = trainable_parameters(feature_model)
        logger.info(f"LoRA trainable params: {tp} || all params: {ap} || trainable%: {100 * tp / ap:.2f}")

        lr_ = backbone_learning_rate
        optim_param_groups.append({'params': feature_model.parameters(), 'lr': lr_})
        checkpoint_model = nn.Sequential(feature_model, linear_classifiers)
    elif peft == "bitfit":
        feature_model = bitfit(feature_model)

        tp, ap = trainable_parameters(feature_model)
        logger.info(f"BitFit trainable params: {tp} || all params: {ap} || trainable%: {100 * tp / ap:.2f}")
        optim_param_groups.append({'params': feature_model.parameters(), 'lr': backbone_learning_rate})
        checkpoint_model = nn.Sequential(feature_model, linear_classifiers)
    else:
        checkpoint_model = linear_classifiers

    # optimizer = torch.optim.SGD(optim_param_groups, momentum=0.9, weight_decay=0.)
    optimizer = torch.optim.AdamW(optim_param_groups, weight_decay=weight_decay)  # zy
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, max_iter, eta_min=0)
    checkpointer = Checkpointer(checkpoint_model, output_dir, optimizer=optimizer, scheduler=scheduler)
    # debug_optimizer_param_shapes(feature_model, linear_classifiers, optimizer)
    start_iter = checkpointer.resume_or_load(classifier_fpath or "", resume=resume).get("iteration", -1) + 1
    if not (resume and checkpointer.has_checkpoint()):  # if not resume
        start_iter = 0

    sampler_type = SamplerType.INFINITE
    train_data_loader, val_data_loader, test_data_loader = make_data_loaders(train_dataset=train_dataset,
                                                                             test_dataset=test_dataset,
                                                                             val_dataset=val_dataset,
                                                                             sampler_type=sampler_type, seed=seed,
                                                                             start_iter=start_iter,
                                                                             batch_size=batch_size,
                                                                             num_workers=num_workers,
                                                                             collate_fn=collate_fn)

    metrics_file_path = os.path.join(output_dir, "results_eval_linear.json")
    val_results_dict, feature_model, linear_classifiers, iteration = eval_linear(
        feature_model=feature_model,
        linear_classifiers=linear_classifiers,
        train_data_loader=train_data_loader,
        val_data_loader=test_data_loader if val_data_loader is None else val_data_loader,
        metrics_file_path=metrics_file_path,
        optimizer=optimizer,
        scheduler=scheduler,
        output_dir=output_dir,
        max_iter=max_iter,
        checkpoint_period=checkpoint_period,
        running_checkpoint_period=checkpoint_period // 2,
        eval_period=eval_period_iterations,
        metric_type=val_metric_type,
        num_of_classes=num_of_classes,
        resume=resume,
        classifier_fpath=classifier_fpath,
        is_multilabel=is_multilabel,
        focal_loss=focal_loss,
        multiview=multiview,
        peft=peft,
    )

    if val_dataset_str is not None:  # retrain model with validation set.
        start_iter = 0
        if shots is None or num_samples is None:  # If few-shot is enabled, keep training set.
            val_dataset = make_dataset(
                dataset_str=val_dataset_str,
                transform=train_transform,
            )
            train_dataset = torch.utils.data.ConcatDataset([train_dataset, val_dataset])

        epoch_length = math.ceil(len(train_dataset) / (batch_size * dist.get_world_size()))
        eval_period_iterations = eval_period_epochs * epoch_length
        checkpoint_period = save_checkpoint_frequency * epoch_length

        train_data_loader = make_data_loader(
            dataset=train_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=True,
            seed=seed,
            sampler_type=sampler_type,
            sampler_advance=start_iter,
            drop_last=False,
            persistent_workers=False,
            collate_fn=collate_fn
        )
        logger.info("Retraining model with combined dataset from train and validation, using the most optimal hp.")
        hyperparameters = extract_hyperparameters_from_model(val_results_dict["best_classifier"]["name"])
        learning_rate, avgpool, block = hyperparameters["lr"], hyperparameters["avgpool"], hyperparameters["blocks"]
        # 微调阶段-线性探测分类疾病
        if not glori:
            linear_classifiers, optim_param_groups = setup_linear_classifiers(
                sample_output=sample_output,
                n_last_blocks_list=block,
                learning_rates=learning_rate,
                avgpools=avgpool,
                num_classes=num_of_classes,
                is_3d=is_3d,
                multiview=multiview,
            )
        else:
            # 复杂下游任务-GLORI模块预测机会性任务
            linear_classifiers, optim_param_groups = setup_glori(
                sample_output=sample_output,
                n_last_blocks_list=block,
                learning_rates=learning_rate,
                avgpools=avgpool,
                num_classes=num_of_classes,
                multiview=multiview,
                decoder_dim=decoder_dim,
                cat_cls=cat_cls,
            )

        output_dir_ = output_dir
        output_dir += os.sep + 'optimal'
        os.makedirs(output_dir, exist_ok=True)

        if feature_model.fine_tune or peft is not None:
            optim_param_groups.append({'params': feature_model.parameters(), 'lr': backbone_learning_rate})

        # optimizer = torch.optim.SGD(optim_param_groups, momentum=0.9, weight_decay=0)
        optimizer = torch.optim.AdamW(optim_param_groups,
                                      weight_decay=weight_decay)  # todo: if peft, this should be with the feature model
        # debug_optimizer_param_shapes(feature_model, linear_classifiers, optimizer)
        max_iter = epochs * epoch_length
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, max_iter, eta_min=0)
        # checkpointer = Checkpointer(linear_classifiers, output_dir, optimizer=optimizer, scheduler=scheduler)  # todo: if peft, this should be with the feature model

        val_results_dict, feature_model, linear_classifiers, iteration = eval_linear(
            feature_model=feature_model,
            linear_classifiers=linear_classifiers,
            train_data_loader=train_data_loader,
            val_data_loader=test_data_loader,
            metrics_file_path=metrics_file_path,
            optimizer=optimizer,
            scheduler=scheduler,
            output_dir=output_dir,
            max_iter=max_iter,
            checkpoint_period=checkpoint_period,
            running_checkpoint_period=checkpoint_period // 2,
            eval_period=eval_period_iterations,
            metric_type=val_metric_type,
            num_of_classes=num_of_classes,
            resume=resume,
            classifier_fpath=output_dir_ + '/model_final.pth',
            is_multilabel=is_multilabel,
            focal_loss=focal_loss,
            multiview=multiview,
        )

    results_dict = {}
    results_dict["best_classifier"] = val_results_dict["best_classifier"]
    logger.info("Test Results Dict " + str(results_dict))

    return results_dict


def run_infer_linear(
        model,
        output_dir,
        test_dataset_str,
        batch_size,
        metric_type,
        num_workers,
        autocast_dtype,
        classifier_fpath=None,
        image_size=224,
        multiview=False,
        glori=False,
        decoder_dim=768,
        cat_cls=False,
):
    seed = 0
    torch.manual_seed(seed)

    if test_dataset_str is None:
        raise ValueError("Test dataset cannot be None")

    eval_transform = make_classification_eval_transform(resize_size=image_size, crop_size=image_size)

    test_dataset = make_dataset(
        dataset_str=test_dataset_str,
        transform=eval_transform,
    )

    num_classes = 2
    print("num_classes",num_classes)
    num_classes = 1 if num_classes == 2 else num_classes
    is_3d = False
    fine_tune = False
    collate_fn = None

    # Load best classifier hyperparameters
    log_json = os.path.join(os.path.dirname(classifier_fpath).rstrip('optimal'), 'results_eval_linear.json')
    with open(log_json, 'r') as f:
        content = f.read().split('\n')[-3]
        data = json.loads(content)
    best_classifier_str = data['best_classifier']['name']
    # best_classifier_str = list(torch.load(classifier_fpath, map_location='cuda:0')['model'].keys())[0].split('.')[1]
    hyperparameters = extract_hyperparameters_from_model(best_classifier_str)
    learning_rate, avgpool, block = hyperparameters["lr"], hyperparameters["avgpool"], hyperparameters["blocks"]

    autocast_ctx = partial(torch.cuda.amp.autocast, enabled=True, dtype=autocast_dtype)
    feature_model = ModelWithIntermediateLayers(model, block[0], autocast_ctx, is_3d=is_3d, fine_tune=fine_tune)

    sample_input = test_dataset[0][0]
    if multiview:
        sample_input = test_dataset[0][0][0]
    sample_input = sample_input.unsqueeze(0).cuda()
    sample_output = feature_model.forward_(sample_input)

    if not glori:
        linear_classifiers, _ = setup_linear_classifiers(
            sample_output=sample_output,
            n_last_blocks_list=block,
            learning_rates=learning_rate,
            avgpools=avgpool,
            num_classes=num_classes,
            is_3d=is_3d,
            multiview=multiview,
        )
    else:
        linear_classifiers, _ = setup_glori(
            sample_output=sample_output,
            n_last_blocks_list=block,
            learning_rates=learning_rate,
            avgpools=avgpool,
            num_classes=num_classes,
            multiview=multiview,
            decoder_dim=decoder_dim,
            cat_cls=cat_cls,
        )

    test_data_loader = make_data_loader(
        dataset=test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        sampler_type=None,
        drop_last=False,
        shuffle=False,
        persistent_workers=False,
        collate_fn=collate_fn,
    )

    checkpointer = Checkpointer(linear_classifiers)
    iteration = checkpointer.load(classifier_fpath).get("iteration")

    metrics_file_path = os.path.join(output_dir, "results_eval_linear.json")

    metric_inputs = evaluate_linear_classifiers_preds(
        feature_model=feature_model,
        linear_classifiers=remove_ddp_wrapper(linear_classifiers),
        data_loader=test_data_loader,
        metrics_file_path=metrics_file_path,
        prefixstring=f"ITER: {iteration} {test_data_loader.dataset.split.value}",
        metric_type=metric_type,
        num_of_classes=num_classes,
        iteration=iteration,
        multiview=multiview,
    )

    if "CheXpert" in test_dataset_str:
        paths = list(test_dataset.labels['Path'])
        preds = metric_inputs['preds'].cpu().numpy()
        class_names = test_dataset.class_names
        df_dict = {'Path': paths}
        df_dict.update({class_names[i]: preds[:, i] for i in range(len(class_names))})
        df = pd.DataFrame(df_dict, )
        df.to_csv(os.path.join(output_dir, 'predictive_probs.csv'), index=False)

    elif 'PLCORotateDet' in test_dataset_str:
        paths = list(test_dataset.df['fpath'])
        preds = metric_inputs['preds'].cpu().numpy()
        df_dict = {'Path': paths, 'Probs': preds}
        df = pd.DataFrame(df_dict, )
        df.to_csv(os.path.join(output_dir, 'predictive_probs.csv'), index=False)
    elif 'PLCO' in test_dataset_str:
        df = test_dataset.df
        preds = metric_inputs['preds'].cpu().numpy()
        df['Probs'] = preds
        df.to_csv(os.path.join(output_dir, 'predictive_probs.csv'), index=False)

    elif 'NLST' in test_dataset_str:
        df = test_dataset.df
        preds = metric_inputs['preds'].cpu().numpy()
        df['Probs'] = preds
        df.to_csv(os.path.join(output_dir, 'predictive_probs.csv'), index=False)

    elif any(_ in test_dataset_str for _ in ['Shenzhen', 'Montgomery', 'JSRT']):
        paths = list(test_dataset.labels['path'])
        preds = metric_inputs['preds'].cpu().numpy()
        df_dict = {'Path': paths}
        if 'ood' in output_dir:
            class_names = test_dataset.class_names
            df_dict.update({class_names[i]: preds[:, i] for i in range(len(class_names))})
        else:
            df_dict.update({'Probs': preds})
        df = pd.DataFrame(df_dict, )
        df.to_csv(os.path.join(output_dir, 'predictive_probs.csv'), index=False)

    else:
        raise NotImplementedError
    return


def run_infer_cxrlt(
        model,
        output_dir,
        test_dataset_str,
        batch_size,
        metric_type,
        num_workers,
        autocast_dtype,
        classifier_fpath=None,
        image_size=224,
        multiview=False,
        glori=False,
        flip=False,
        decoder_dim=768,
        cat_cls=False,
):
    seed = 0
    torch.manual_seed(seed)

    if test_dataset_str is None:
        raise ValueError("Test dataset cannot be None")

    eval_transform = make_classification_eval_transform(resize_size=image_size, crop_size=image_size, flip=flip)

    test_dataset = make_dataset(
        dataset_str=test_dataset_str,
        transform=eval_transform,
    )

    num_classes = 2
    print("num_classes", num_classes)
    num_classes = 1 if num_classes == 2 else num_classes
    is_3d = False
    fine_tune = False
    collate_fn = None

    # Load best classifier hyperparameters
    if os.path.dirname(classifier_fpath).split('/')[-1] == 'optimal':
        log_json = os.path.join(os.path.dirname(classifier_fpath).rstrip('optimal'), 'results_eval_linear.json')
    else:
        log_json = os.path.join(os.path.dirname(classifier_fpath), 'results_eval_linear.json')
    with open(log_json, 'r') as f:
        content = f.read().split('\n')[-3]
        data = json.loads(content)
    best_classifier_str = data['best_classifier']['name']
    # best_classifier_str = list(torch.load(classifier_fpath, map_location='cuda:0')['model'].keys())[0].split('.')[1]
    hyperparameters = extract_hyperparameters_from_model(best_classifier_str)
    learning_rate, avgpool, block = hyperparameters["lr"], hyperparameters["avgpool"], hyperparameters["blocks"]
    # learning_rate, avgpool, block = [1e-4], [False], [4]  # zy

    autocast_ctx = partial(torch.cuda.amp.autocast, enabled=True, dtype=autocast_dtype)  # zy
    feature_model = ModelWithIntermediateLayers(model, block[0], autocast_ctx, is_3d=is_3d, fine_tune=fine_tune)

    sample_input = test_dataset[0][0]
    if multiview:
        sample_input = test_dataset[0][0][0]
    sample_input = sample_input.unsqueeze(0).cuda()
    sample_output = feature_model.forward_(sample_input)
    # 线性探测-胸部疾病分类
    if not glori:
        linear_classifiers, _ = setup_linear_classifiers(
            sample_output=sample_output,
            n_last_blocks_list=block,
            learning_rates=learning_rate,
            avgpools=avgpool,
            num_classes=num_classes,
            is_3d=is_3d,
            multiview=multiview,
        )
    # 扩展适配阶段-机会性任务
    else:
        linear_classifiers, _ = setup_glori(
            sample_output=sample_output,
            n_last_blocks_list=block,
            learning_rates=learning_rate,
            avgpools=avgpool,
            num_classes=num_classes,
            multiview=multiview,
            decoder_dim=decoder_dim,
            cat_cls=cat_cls,
        )

    test_data_loader = make_data_loader(
        dataset=test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        sampler_type=None,
        drop_last=False,
        shuffle=False,
        persistent_workers=False,
        collate_fn=collate_fn,
    )

    checkpointer = Checkpointer(linear_classifiers)
    iteration = checkpointer.load(classifier_fpath).get("iteration")

    metrics_file_path = os.path.join(output_dir, "results_eval_linear.json")

    # get predictions
    metric_inputs = evaluate_linear_classifiers_preds(
        feature_model=feature_model,
        linear_classifiers=remove_ddp_wrapper(linear_classifiers),
        data_loader=test_data_loader,
        metrics_file_path=metrics_file_path,
        prefixstring=f"ITER: {iteration} {test_data_loader.dataset.split.value}",
        metric_type=metric_type,
        num_of_classes=num_classes,
        iteration=iteration,
        multiview=multiview,
    )

    # save csv
    dicom_ids = list(test_dataset.labels['dicom_id'])
    preds = metric_inputs['preds'].cpu().numpy()
    class_names = test_dataset.class_names
    df_dict = {'dicom_id': dicom_ids}
    df_dict.update({class_names[i]: preds[:, i] for i in range(len(class_names))})
    df = pd.DataFrame(df_dict, )
    df.to_csv(os.path.join(output_dir, 'cxrlt_submission.csv'), index=False)
    return


def main(args):
    model, autocast_dtype = setup_and_build_model(args)
    run = partial(run_eval_linear,
                  model=model,
                  train_dataset_str=args.train_dataset_str,
                  val_dataset_str=args.val_dataset_str,
                  test_dataset_str=args.test_dataset_str,
                  batch_size=args.batch_size,
                  epochs=args.epochs,
                  val_epochs=args.val_epochs,
                  epoch_length=args.epoch_length,
                  num_workers=args.num_workers,
                  save_checkpoint_frequency=args.save_checkpoint_frequency,
                  eval_period_epochs=args.eval_period_epochs,
                  learning_rates=args.learning_rates,
                  backbone_learning_rate=args.backbone_learning_rate,
                  n_last_blocks_list=args.n_last_blocks,
                  avgpools=args.avgpools,
                  autocast_dtype=autocast_dtype,
                  resume=not args.no_resume,
                  classifier_fpath=args.classifier_fpath,
                  val_metric_type=args.val_metric_type,
                  fine_tune=args.fine_tune,
                  backbone=args.backbone,
                  peft=args.peft,
                  image_size=args.image_size,
                  num_samples=args.num_samples,
                  focal_loss=args.focal_loss,
                  multiview=args.multiview,
                  glori=args.glori,
                  weight_decay=args.weight_decay,
                  rot90_train=args.rot90_train,
                  decoder_dim=args.decoder_dim,
                  cat_cls=args.cat_cls,
                  )

    run_infer = partial(
        run_infer_linear,
        model=model,
        output_dir=args.output_dir,
        test_dataset_str=args.test_dataset_str,
        batch_size=args.batch_size,
        metric_type=args.val_metric_type,
        num_workers=args.num_workers,
        autocast_dtype=autocast_dtype,
        classifier_fpath=args.classifier_fpath,
        image_size=args.image_size,
        multiview=args.multiview,
        glori=args.glori,
        decoder_dim=args.decoder_dim,
        cat_cls=args.cat_cls,
    )

    run_cxrlt = partial(
        run_infer_cxrlt,
        model=model,
        output_dir=args.output_dir,
        test_dataset_str=args.test_dataset_str,
        batch_size=args.batch_size,
        metric_type=args.val_metric_type,
        num_workers=args.num_workers,
        autocast_dtype=autocast_dtype,
        classifier_fpath=args.classifier_fpath,
        image_size=args.image_size,
        multiview=args.multiview,
        glori=args.glori,
        flip=args.flip,
        decoder_dim=args.decoder_dim,
        cat_cls=args.cat_cls,
    )

    if not args.infer:
        if args.shots != None:
            for shot in args.shots:
                fs_output_dir = args.output_dir + os.sep + f"shots-{shot}"
                os.makedirs(fs_output_dir, exist_ok=True)
                run(shots=shot, output_dir=fs_output_dir)
        else:
            run(shots=args.shots, output_dir=args.output_dir, )
    else:
        if 'CXRLT' not in args.test_dataset_str:
            run_infer()
        else:
            run_cxrlt()
    return 0


if __name__ == "__main__":
    description = "DINOv2 linear evaluation"
    args_parser = get_args_parser(description=description)
    args = args_parser.parse_args()
    sys.exit(main(args))
