#!/usr/bin/python3

"""
# -*- coding: utf-8 -*-

# @Time     : 2020/8/11 13:10
# @File     : trainer.py

"""
import os
import torch
from tqdm import tqdm
import math
import time
import random
import glob

from gigadetect.core import losser, evaluater
from gigadetect.models import model_loader
from gigadetect.models.dummy_model import DummyModel
from gigadetect.models.yolov5.yolo import Model
from gigadetect.datasets import coco as coco_dataset
from gigadetect.utils.logger import GIGA_DETECT_LOGGER
from gigadetect.utils import misc
from torch import optim
from torch.optim import lr_scheduler
import numpy as np
import torch.nn.functional as F


def get_optimizer(cfg, model, weight_decay):

    pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
    for k, v in model.named_parameters():
        if v.requires_grad:
            if '.bias' in k:
                pg2.append(v)  # biases
            elif '.weight' in k and '.bn' not in k:
                pg1.append(v)  # apply weight decay
            else:
                pg0.append(v)  # all else

    if cfg.SOLVER.OPTIMIZER == 'Adam':
        optimizer = optim.Adam(pg0, lr=cfg.SOLVER.BASE_LR)
    else:
        optimizer = optim.SGD(pg0, lr=cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM, nesterov=True)
    optimizer.add_param_group({'params': pg1, 'weight_decay': weight_decay})  # add pg1 with weight_decay
    optimizer.add_param_group({'params': pg2})  # add pg2 (biases)
    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    GIGA_DETECT_LOGGER.info('Optimizer groups: %g .bias, %g conv.weight, %g other' % (len(pg2), len(pg1), len(pg0)))
    del pg0, pg1, pg2

    return optimizer


def learning_rate_function(epochs):
    lf = lambda x: (((1 + math.cos(x * math.pi / epochs)) / 2) ** 1.0) * 0.9 + 0.1  # cosine
    return lf


def get_scheduler(epochs, optimizer):
    lf = learning_rate_function(epochs)
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    return scheduler


def attempt_burn_in_train(cfg, epoch, ni, num_burn_in, nominal_batch_size, batch_size, optimizer, lf):
        xi = [0, num_burn_in]  # x interp
        # model.gr = np.interp(ni, xi, [0.0, 1.0])  # giou loss ratio (obj_loss = 1.0 or giou)
        accumulate = max(1, np.interp(ni, xi, [1, nominal_batch_size / batch_size]).round())
        for j, x in enumerate(optimizer.param_groups):
            # bias lr falls from 0.1 to `BASE_LR`, all other lrs rise from 0.0 to `BASE_LR`
            x['lr'] = np.interp(ni, xi, [0.1 if j == 2 else 0.0, x['initial_lr'] * lf(epoch)])
            if 'momentum' in x:
                x['momentum'] = np.interp(ni, xi, [0.9, cfg.SOLVER.MOMENTUM])


def multi_scale_pre_process(imgs, imgsz, grid_size):
    sz = random.randrange(imgsz * 0.5, imgsz * 1.5 + grid_size) // grid_size * grid_size  # size
    sf = sz / max(imgs.shape[2:])  # scale factor
    if sf != 1:
        ns = [math.ceil(x * sf / grid_size) * grid_size for x in imgs.shape[2:]]  # new shape (stretched to gs-multiple)
        imgs = F.interpolate(imgs, size=ns, mode='bilinear', align_corners=False)
    return imgs


def set_progress_bar_description(mloss, loss_items, i, epoch, epochs, targets, imgs, pbar):
    mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
    mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)  # (GB)
    s = ('%10s' * 2 + '%10.4g' * 6) % (
        '%g/%g' % (epoch, epochs - 1), mem, *mloss, targets.shape[0], imgs.shape[-1])
    pbar.set_description(s)
    return s


def update_train_set_image_weights(train_dataset, model, map_per_class, num_classes):
    # TODO: test this logic
    return
    if train_dataset.image_weights:
        class_weights = misc.labels_to_class_weights(train_dataset.labels, num_classes).to(
            device)  # attach class weights

        w = class_weights.cpu().numpy() * (1 - map_per_class) ** 2  # class weights
        image_weights = misc.labels_to_image_weights(train_dataset.labels, nc=num_classes, class_weights=w)
        train_dataset.indices = random.choices(range(train_dataset.n), weights=image_weights, k=train_dataset.n)  # rand weighted idx


def write_result_file(cfg, results, desc):
    save_dir = cfg.OUTPUT.ROOT_DIR
    results_file = os.path.join(save_dir, 'results.txt')
    with open(results_file, 'a') as f:
        f.write(desc + '%10.4g' * 7 % results + '\n')  # P, R, mAP, F1, test_losses=(GIoU, obj, cls)


def save_model(cfg, epoch, best_fitness, model, optimizer, epochs, fitness_score, num_classes):
    save_dir = cfg.OUTPUT.ROOT_DIR

    final_epoch = epoch + 1 == epochs
    ckpt = {'epoch': epoch,
            'best_fitness': best_fitness,
            'arch': cfg.MODEL.ARCH,
            'num_classes': num_classes,
            'training_results': None,
            'state_dict': model.state_dict(),
            'optimizer': None if final_epoch else optimizer.state_dict()}

    # Save last, best and delete
    last_path = os.path.join(save_dir, 'last.pth')
    best_path = os.path.join(save_dir, 'best.pth')
    GIGA_DETECT_LOGGER.info("[Train] save latest model to {}", last_path)
    torch.save(ckpt, last_path)
    if (best_fitness == fitness_score) and not final_epoch:
        GIGA_DETECT_LOGGER.info("[Train] save best model to {}", best_path)
        torch.save(ckpt, best_path)

    del ckpt


def epoch_evaluate(model, test_data_loader, cfg, epoch, epochs, batch_size, skip_evaluate=False):
    final_epoch = epoch + 1 == epochs
    if not skip_evaluate or final_epoch:  # Calculate mAP
        results, map_per_class, times = evaluater.evaluate(cfg,
                                                    model,
                                                    data_loader=test_data_loader,
                                                    half=True,
                                                    training=True,
                                                    batch_size=batch_size
                                                    )
        return results, map_per_class, times
    return None, None, None


def process_class_frequency(train_dataset):
    # TODO: add logic of class frequency
    labels = np.concatenate(train_dataset.labels, 0)
    c = torch.tensor(labels[:, 0])  # classes
    # cf = torch.bincount(c.long(), minlength=nc) + 1.
    # model._initialize_biases(cf.to(device))


def load_pretrained_model(cfg, device, cfg_file_path, model, optimizer, epochs):
    weights_path = cfg.MODEL.WEIGHTS
    best_fitness = 0.0
    start_epoch = 0
    if weights_path.endswith('.pth'):  # pytorch format
        ckpt = torch.load(weights_path, map_location=device)  # load checkpoint
        GIGA_DETECT_LOGGER.info('Load pretrained moedl information from {}', weights_path)
        # load model
        try:
            state_dict = ckpt['state_dict']  # to FP32, filter
            model_loader.load_pretrained_state_dict(model, state_dict, device)
            GIGA_DETECT_LOGGER.info('Transferred %g/%g items from %s' % (len(model.state_dict()), len(state_dict), weights))
        except KeyError as e:
            s = "%s is not compatible with %s. This may be due to model differences or %s may be out of date. " \
                "Please delete or update %s and try again, or use --weights '' to train from scratch." \
                % (weights_path, cfg_file_path, weights_path, weights_path)
            raise KeyError(s) from e

        # load optimizer
        if ckpt.get('optimizer') is not None:
            # optimizer.load_state_dict(ckpt['optimizer'])
            # best_fitness = ckpt['best_fitness']
            pass

        # load results
        if ckpt.get('training_results') is not None:
            pass
            # TODO  record result form 

        # epochs
        if ckpt.get('epoch') is not None:
            start_epoch = ckpt['epoch'] + 1
            if epochs < start_epoch:
                GIGA_DETECT_LOGGER.info('%s has been trained for %g epochs. Fine-tuning for %g additional epochs.' %
                    (weights_path, ckpt['epoch'], epochs))
                epochs += ckpt['epoch']  # finetune additional epochs

        del ckpt
    else:
        GIGA_DETECT_LOGGER.warning("[Train] The input pretrained models {} seems not a valid PyTorch model, skipped...", weights_path)
    
    return best_fitness, start_epoch, epochs


def train(cfg):
    device = misc.get_target_device(cfg.MODEL.CUDA_DEVICE, apex=False, batch_size=cfg.SOLVER.BATCH_SIZE)

    # Configure
    batch_size = cfg.SOLVER.BATCH_SIZE
    num_classes = 1 if cfg.DATASETS.SINGLE_CLASS else cfg.DATASETS.NUM_CLASSES
    assert num_classes == len(cfg.DATASETS.CLASS_NAMES), 'names are inconsistant with num_classes'

    cfg.defrost()
    cfg.MODEL.LOSS.CLS_GAIN *= num_classes / 80.   # scale coco-tuned `cfg.MODEL.LOSS.CLS_GAIN` to current dataset

    # initialize seeds
    misc.init_seeds(1)

    # Create model
    model = model_loader.initialize_yolov5_model(cfg.MODEL.CFG_FILE_PATH, cfg.DATASETS.NUM_CLASSES, device)

    # Optimizer
    nominal_batch_size = 64
    accumulate = max(round(nominal_batch_size / batch_size), 1)  # accumulate loss before optimizing
    weight_decay = cfg.SOLVER.WEIGHT_DECAY * batch_size * accumulate / nominal_batch_size  # scale weight_decay
    optimizer = get_optimizer(cfg, model, weight_decay)

    # scheduler
    max_epoch = cfg.SOLVER.MAX_EPOCH
    scheduler = get_scheduler(max_epoch, optimizer)

    # Load Model
    best_fitness, start_epoch, max_epoch = load_pretrained_model(cfg, device, cfg.MODEL.CFG_FILE_PATH, model, optimizer, max_epoch)

    # Mixed precision training https://github.com/NVIDIA/apex
    # TODO: add mixed_precision training

    # https://discuss.pytorch.org/t/a-problem-occured-when-resuming-an-optimizer/28822
    # plot_lr_scheduler(optimizer, scheduler, epochs)

    # Initialize distributed training
    # TODO: add distributed training

    # Image sizes
    grid_size = model.stride.max()  # grid size (max stride)
    train_image_size = misc.check_img_size(cfg.INPUT.IMAGE_SIZE_TRAIN[0], s=grid_size)
    test_image_size = misc.check_img_size(cfg.INPUT.IMAGE_SIZE_TEST[0], s=grid_size)
    GIGA_DETECT_LOGGER.info('Image sizes %g train, %g test' % (train_image_size, test_image_size))

    # Train loader
    train_data_loader, train_dataset = coco_dataset.create_data_loader(cfg, cfg.DATASETS.TRAIN_ROOT, train_image_size, batch_size, grid_size, cfg.DATASETS.SINGLE_CLASS,
                                                                       augment=True,
                                                                       cache=cfg.DATASETS.CACHE_IMAGES, rect=cfg.DATASETS.RECT_TRAINING)
    max_label_id = np.concatenate(train_dataset.labels, 0)[:, 0].max()  # max label class
    assert max_label_id < num_classes, 'Label class %g exceeds nc=%g in %s.' % (max_label_id, num_classes, cfg.MODEL.CFG_FILE_PATH)


    # Test loader
    test_data_loader, _ = coco_dataset.create_data_loader(cfg, cfg.DATASETS.VAL_ROOT, test_image_size, batch_size, grid_size, cfg.DATASETS.SINGLE_CLASS,
                                                          augment=False,
                                                          cache=cfg.DATASETS.CACHE_IMAGES, rect=True)

    # Class frequency
    process_class_frequency(train_dataset)

    # Check anchors TODO: add noautoanchor check

    # Exponential moving average

    # Start training
    t0 = time.time()
    num_batches = len(train_data_loader)  # number of batches
    num_burn_in = max(3 * num_batches, 1e3)  # burn-in iterations, max(3 epochs, 1k iterations)
    map_per_class = np.zeros(num_classes)  # mAP per class
    results = (0, 0, 0, 0, 0, 0, 0)  # 'P', 'R', 'mAP', 'F1', 'val GIoU', 'val Objectness', 'val Classification'
    scheduler.last_epoch = start_epoch - 1  # do not move
    lf = learning_rate_function(cfg.SOLVER.MAX_EPOCH)
    GIGA_DETECT_LOGGER.info('Using %g dataloader workers' % train_data_loader.num_workers)
    GIGA_DETECT_LOGGER.info('Starting training for %g epochs...' % max_epoch)


    # iteragte epoch 
    for epoch in range(start_epoch, max_epoch):
        model.train()

        # Update image weights (optional)
        update_train_set_image_weights(train_dataset, model, map_per_class, num_classes)

        # Update mosaic border
        # b = int(random.uniform(0.25 * imgsz, 0.75 * imgsz + gs) // gs * gs)
        # dataset.mosaic_border = [b - imgsz, -b]  # height, width borders

        mloss = torch.zeros(4, device=device)  # mean losses
        print(('\n' + '%10s' * 8) % ('Epoch', 'gpu_mem', 'GIoU', 'obj', 'cls', 'total', 'targets', 'img_size'))
        desc = ''
        pbar = tqdm(enumerate(train_data_loader), total=num_batches)  # progress bar
        for i, (imgs, targets, paths, _) in pbar:  # batch -------------------------------------------------------------
            ni = i + num_batches * epoch  # number integrated batches (since train start)
            imgs = imgs.to(device).float() / 255.0  # uint8 to float32, 0 - 255 to 0.0 - 1.0

            # Burn-in
            attempt_burn_in_train(cfg, epoch, ni, num_burn_in, nominal_batch_size, batch_size, optimizer, lf)
            
            # Multi-scale
            if cfg.INPUT.MULTI_SCALE:
                imgs = multi_scale_pre_process(imgs, cfg.INPUT.IMAGE_SIZE_TRAIN[0], grid_size)

            # Forward
            pred = model(imgs)

            # Loss
            loss, loss_items = losser.compute_loss(cfg, pred, targets.to(device), model)
            if not torch.isfinite(loss):
                print('WARNING: non-finite loss, ending training ', loss_items)
                return results

            # Backward
            loss.backward()
            # TODO: add mixed_precision training

            # Optimize
            if ni % accumulate == 0:
                optimizer.step()
                optimizer.zero_grad()

            # Print
            desc = set_progress_bar_description(mloss, loss_items, i, epoch, max_epoch, targets, imgs, pbar)

            # TODO: add plot logic
            # end batch ------------------------------------------------------------------------------------------------

        # Scheduler
        scheduler.step()

        # evaluate
        results, map_per_class, times = epoch_evaluate(model, test_data_loader, cfg, epoch, max_epoch, batch_size, cfg.SOLVER.SKIP_EVALUATE)

        # Write result file
        write_result_file(cfg, results, desc)

        # TODO: add logic of tensorboard

        # Update best mAP
        fitness_score = misc.fitness(np.array(results).reshape(1, -1))  # fitness_i = weighted combination of [P, R, mAP, F1]
        if fitness_score > best_fitness:
            best_fitness = fitness_score

        # Save model
        save_model(cfg, epoch, best_fitness, model, optimizer, max_epoch, fitness_score, num_classes)

        # end epoch ----------------------------------------------------------------------------------------------------
    # end training

    # Strip optimizers
    # TODO: strip optimizers

    # TODO: plot result


    # Finish
    torch.cuda.empty_cache()
    return results

