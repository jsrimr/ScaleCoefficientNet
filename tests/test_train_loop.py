import argparse
import random

import torch
import torch.nn.functional as F

from ofa.imagenet_classification.elastic_nn.networks import OFAMobileNetV3
from ofa.imagenet_classification.networks import MobileNetV3Large

from ofa.utils import AverageMeter, cross_entropy_loss_with_soft_target
from ofa.utils import list_mean, subset_mean, MyRandomResizedCrop
from train_loop import DistributedRunConfig, DistributedRunManager


def init_args(args):
    # task 가 여러개 있네? once-for-all 은
    if args.task == 'kernel':
        args.path = 'exp/normal2kernel'
        args.dynamic_batch_size = 1
        args.n_epochs = 120
        args.base_lr = 3e-2
        args.warmup_epochs = 5
        args.warmup_lr = -1
        args.ks_list = '3,5,7'
        args.expand_list = '6'
        args.depth_list = '4'
    elif args.task == 'depth':
        args.path = 'exp/kernel2kernel_depth/phase%d' % args.phase
        args.dynamic_batch_size = 2
        if args.phase == 1:
            args.n_epochs = 25
            args.base_lr = 2.5e-3
            args.warmup_epochs = 0
            args.warmup_lr = -1
            args.ks_list = '3,5,7'
            args.expand_list = '6'
            args.depth_list = '3,4'
        else:
            args.n_epochs = 120
            args.base_lr = 7.5e-3
            args.warmup_epochs = 5
            args.warmup_lr = -1
            args.ks_list = '3,5,7'
            args.expand_list = '6'
            args.depth_list = '2,3,4'
    elif args.task == 'expand':
        args.path = 'exp/kernel_depth2kernel_depth_width/phase%d' % args.phase
        args.dynamic_batch_size = 4
        if args.phase == 1:
            args.n_epochs = 25
            args.base_lr = 2.5e-3
            args.warmup_epochs = 0
            args.warmup_lr = -1
            args.ks_list = '3,5,7'
            args.expand_list = '4,6'
            args.depth_list = '2,3,4'
        else:
            args.n_epochs = 120
            args.base_lr = 7.5e-3
            args.warmup_epochs = 5
            args.warmup_lr = -1
            args.ks_list = '3,5,7'
            args.expand_list = '3,4,6'
            args.depth_list = '2,3,4'
    else:
        raise NotImplementedError
    args.manual_seed = 0

    args.lr_schedule_type = 'cosine'

    args.base_batch_size = 64
    args.valid_size = 10000

    args.opt_type = 'sgd'
    args.momentum = 0.9
    args.no_nesterov = False
    args.weight_decay = 3e-5
    args.label_smoothing = 0.1
    args.no_decay_keys = 'bn#bias'
    args.fp16_allreduce = False

    args.model_init = 'he_fout'
    args.validation_frequency = 1
    args.print_frequency = 10

    args.n_worker = 8
    args.resize_scale = 0.08
    args.distort_color = 'tf'
    args.image_size = [128,160,192,224]
    args.continuous_size = True
    args.not_sync_distributed_image_size = False

    args.bn_momentum = 0.1
    args.bn_eps = 1e-5
    args.dropout = 0.1
    args.base_stage_width = 'proxyless'

    args.width_mult_list = '1.0'
    args.dy_conv_scaling_mode = 1
    args.independent_distributed_sampling = False

    args.kd_ratio = 1.0
    args.kd_type = 'ce'

    return args

def get_run_config_and_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='depth', choices=[
        'kernel', 'depth', 'expand',
    ])
    parser.add_argument('--phase', type=int, default=1, choices=[1, 2])
    args = parser.parse_args(['--task', 'depth', '--phase', '1'])
    args = init_args(args)

    run_config = DistributedRunConfig(**args.__dict__)

    # build net from args
    args.width_mult_list = [float(width_mult) for width_mult in args.width_mult_list.split(',')]
    args.ks_list = [int(ks) for ks in args.ks_list.split(',')]
    args.expand_list = [int(e) for e in args.expand_list.split(',')]
    args.depth_list = [int(d) for d in args.depth_list.split(',')]

    args.width_mult_list = args.width_mult_list[0] if len(args.width_mult_list) == 1 else args.width_mult_list

    return run_config, args

def get_run_manager_and_args():
    run_config, args = get_run_config_and_args()

    net = OFAMobileNetV3(
        n_classes=run_config.data_provider.n_classes, bn_param=(args.bn_momentum, args.bn_eps),
        dropout_rate=args.dropout, base_stage_width=args.base_stage_width, width_mult=args.width_mult_list,
        ks_list=args.ks_list, expand_ratio_list=args.expand_list, depth_list=args.depth_list
    )

    distributed_run_manager = DistributedRunManager(
        args.path, net, run_config
    )
    distributed_run_manager.save_config()

    return distributed_run_manager, args


def test_train_loop():
    distributed_run_manager, args = get_run_manager_and_args()
    # train_loss, (train_top1, train_top5) = distributed_run_manager.train_one_epoch(args, epoch=0)
    distributed_run_manager.train_one_epoch(args, epoch=0)
    # print(f"top1 acc = {train_top1}")


def test_save_model():
    pass


def test_load_model():
    pass


def test_tpu_trainer():
    pass


def test_BN_recalibration():
    pass


def test_logging():
    pass