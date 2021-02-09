"""
Code from once-for-all/ofa/imagenet_classification/run_manager/run_config.py
"""

import json
import os
import random
import time

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.debug.metrics as met
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.utils.utils as xu
from tqdm import tqdm

from ofa.utils import MyRandomResizedCrop
from ofa.utils import calc_learning_rate, build_optimizer
from ofa.utils import cross_entropy_with_label_smoothing, cross_entropy_loss_with_soft_target, write_log, init_models
from ofa.utils import get_net_info, accuracy, AverageMeter, mix_labels, mix_images

__all__ = ['RunManager']
SERIAL_EXEC = xmp.MpSerialExecutor()

class RunConfig:

    def __init__(self, n_epochs, init_lr, lr_schedule_type, lr_schedule_param,
                 dataset, train_batch_size, test_batch_size, valid_size,
                 opt_type, opt_param, weight_decay, label_smoothing, no_decay_keys,
                 mixup_alpha, model_init, validation_frequency, print_frequency):
        self.n_epochs = n_epochs
        self.init_lr = init_lr
        self.lr_schedule_type = lr_schedule_type
        self.lr_schedule_param = lr_schedule_param

        self.dataset = dataset
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.valid_size = valid_size

        self.opt_type = opt_type
        self.opt_param = opt_param
        self.weight_decay = weight_decay
        self.label_smoothing = label_smoothing
        self.no_decay_keys = no_decay_keys  # todo: 이게 뭘까

        self.mixup_alpha = mixup_alpha  # todo : 호 이게 그 유명한 mixup 기법인가

        self.model_init = model_init
        self.validation_frequency = validation_frequency
        self.print_frequency = print_frequency

    @property
    def config(self):
        config = {}
        for key in self.__dict__:
            if not key.startswith('_'):
                config[key] = self.__dict__[key]
        return config

    def copy(self):
        return RunConfig(**self.config)

    """ learning rate """

    def adjust_learning_rate(self, optimizer, epoch, batch=0, nBatch=None):
        """ adjust learning of a given optimizer and return the new learning rate """
        new_lr = calc_learning_rate(epoch, self.init_lr, self.n_epochs, batch, nBatch, self.lr_schedule_type)
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr
        return new_lr

    def warmup_adjust_learning_rate(self, optimizer, T_total, nBatch, epoch, batch=0, warmup_lr=0):
        T_cur = epoch * nBatch + batch + 1
        new_lr = T_cur / T_total * (self.init_lr - warmup_lr) + warmup_lr
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr
        return new_lr

    """ data provider """

    @property
    def data_provider(self):
        raise NotImplementedError

    @property
    def train_loader(self):
        return self.data_provider.train

    @property
    def valid_loader(self):
        return self.data_provider.valid

    @property
    def test_loader(self):
        return self.data_provider.test

    def random_sub_train_loader(self, n_images, batch_size, num_worker=None, num_replicas=None, rank=None):
        return self.data_provider.build_sub_train_loader(n_images, batch_size, num_worker, num_replicas, rank)

    """ optimizer """

    def build_optimizer(self, net_params):
        return build_optimizer(net_params,
                               self.opt_type, self.opt_param, self.init_lr, self.weight_decay, self.no_decay_keys)


from tpu_data_provider import RandomSizedCocoDataProvider


class DistributedRunConfig(RunConfig):

    def __init__(self, n_epochs=150, init_lr=0.05, lr_schedule_type='cosine', lr_schedule_param=None,
                 dataset='CoCoTPU', train_batch_size=64, test_batch_size=64, valid_size=None,
                 opt_type='sgd', opt_param=None, weight_decay=4e-5, label_smoothing=0.1, no_decay_keys=None,
                 mixup_alpha=None, model_init='he_fout', validation_frequency=1, print_frequency=10,
                 n_worker=8, resize_scale=0.08, distort_color='tf', image_size=224,
                 **kwargs):
        super().__init__(
            n_epochs, init_lr, lr_schedule_type, lr_schedule_param,
            dataset, train_batch_size, test_batch_size, valid_size,
            opt_type, opt_param, weight_decay, label_smoothing, no_decay_keys,
            mixup_alpha, model_init, validation_frequency, print_frequency
        )

        # self._num_replicas = kwargs['num_replicas']
        # self._rank = kwargs['rank']
        self.n_worker = n_worker
        self.resize_scale = resize_scale
        self.distort_color = distort_color
        self.image_size = image_size

    @property
    def data_provider(self):
        if self.__dict__.get('_data_provider', None) is None:
            if self.dataset == RandomSizedCocoDataProvider.name():
                DataProviderClass = RandomSizedCocoDataProvider
            else:
                raise NotImplementedError

            # todo : distort_color 를 "tf" 로 그대로 놔두는게 맞나?
            self.__dict__['_data_provider'] = DataProviderClass(
                train_batch_size=self.train_batch_size, test_batch_size=self.test_batch_size,
                valid_size=self.valid_size, n_worker=self.n_worker, resize_scale=self.resize_scale,
                distort_color=self.distort_color, image_size=self.image_size,
                # num_replicas=self._num_replicas, rank=self._rank,
            )
        return self.__dict__['_data_provider']


class RunManager:

    def __init__(self, path, net, run_config, init=True, measure_latency=None, no_gpu=False):

        self.path = path
        self.net = net
        self.run_config = run_config

        self.best_acc = 0.0
        self.start_epoch = 0

        os.makedirs(self.path, exist_ok=True)

        self.device = xm.xla_device()
        self.net = xmp.MpModelWrapper(self.net).to(self.device)

        # initialize model (default)
        if init:
            init_models(self.net, run_config.model_init)

        # net info
        net_info = get_net_info(self.net, self.run_config.data_provider.data_shape, measure_latency, True)
        with open('%s/net_info.txt' % self.path, 'w') as fout:
            fout.write(json.dumps(net_info, indent=4) + '\n')
            # noinspection PyBroadException
            try:
                fout.write(self.network.module_str + '\n')
            except Exception:
                pass
            fout.write('%s\n' % self.run_config.data_provider.train.dataset.transform)
            fout.write('%s\n' % self.run_config.data_provider.test.dataset.transform)
            fout.write('%s\n' % self.network)

        # criterion
        if isinstance(self.run_config.mixup_alpha, float):
            self.train_criterion = cross_entropy_loss_with_soft_target
        elif self.run_config.label_smoothing > 0:
            self.train_criterion = lambda pred, target: \
                cross_entropy_with_label_smoothing(pred, target, self.run_config.label_smoothing)
        else:
            self.train_criterion = nn.CrossEntropyLoss()
        self.test_criterion = nn.CrossEntropyLoss()

        # optimizer
        if self.run_config.no_decay_keys:
            keys = self.run_config.no_decay_keys.split('#')
            net_params = [
                self.net.get_parameters(keys, mode='exclude'),  # parameters with weight decay
                self.net.get_parameters(keys, mode='include'),  # parameters without weight decay
            ]
        else:
            # noinspection PyBroadException
            try:
                net_params = self.network.weight_parameters()
            except Exception:
                net_params = []
                for param in self.network.parameters():
                    if param.requires_grad:
                        net_params.append(param)
        self.optimizer = self.run_config.build_optimizer(net_params)

    """ save path and log path """

    """ save path and log path """

    @property
    def save_path(self):
        if self.__dict__.get('_save_path', None) is None:
            save_path = os.path.join(self.path, 'checkpoint')
            os.makedirs(save_path, exist_ok=True)
            self.__dict__['_save_path'] = save_path
        return self.__dict__['_save_path']

    @property
    def logs_path(self):
        if self.__dict__.get('_logs_path', None) is None:
            logs_path = os.path.join(self.path, 'logs')
            os.makedirs(logs_path, exist_ok=True)
            self.__dict__['_logs_path'] = logs_path
        return self.__dict__['_logs_path']

    @property
    def network(self):
        return self.net.module if isinstance(self.net, nn.DataParallel) else self.net

    def write_log(self, log_str, prefix='valid', should_print=True, mode='a'):
        write_log(self.logs_path, log_str, prefix, should_print, mode)

    """ save and load models """

    def save_model(self, checkpoint=None, is_best=False, model_name=None):
        if checkpoint is None:
            checkpoint = {'state_dict': self.network.state_dict()}

        if model_name is None:
            model_name = 'checkpoint.pth.tar'

        checkpoint['dataset'] = self.run_config.dataset  # add `dataset` info to the checkpoint
        latest_fname = os.path.join(self.save_path, 'latest.txt')
        model_path = os.path.join(self.save_path, model_name)
        with open(latest_fname, 'w') as fout:
            fout.write(model_path + '\n')
        torch.save(checkpoint, model_path)

        if is_best:
            best_path = os.path.join(self.save_path, 'model_best.pth.tar')
            torch.save({'state_dict': checkpoint['state_dict']}, best_path)

    def load_model(self, model_fname=None):
        latest_fname = os.path.join(self.save_path, 'latest.txt')
        if model_fname is None and os.path.exists(latest_fname):
            with open(latest_fname, 'r') as fin:
                model_fname = fin.readline()
                if model_fname[-1] == '\n':
                    model_fname = model_fname[:-1]
        # noinspection PyBroadException
        try:
            if model_fname is None or not os.path.exists(model_fname):
                model_fname = '%s/checkpoint.pth.tar' % self.save_path
                with open(latest_fname, 'w') as fout:
                    fout.write(model_fname + '\n')
            print("=> loading checkpoint '{}'".format(model_fname))
            checkpoint = torch.load(model_fname, map_location='cpu')
        except Exception:
            print('fail to load checkpoint from %s' % self.save_path)
            return {}

        self.network.load_state_dict(checkpoint['state_dict'])
        if 'epoch' in checkpoint:
            self.start_epoch = checkpoint['epoch'] + 1
        if 'best_acc' in checkpoint:
            self.best_acc = checkpoint['best_acc']
        if 'optimizer' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer'])

        print("=> loaded checkpoint '{}'".format(model_fname))
        return checkpoint

    def save_config(self, extra_run_config=None, extra_net_config=None):
        """ dump run_config and net_config to the model_folder """
        run_save_path = os.path.join(self.path, 'run.config')
        if not os.path.isfile(run_save_path):
            run_config = self.run_config.config
            if extra_run_config is not None:
                run_config.update(extra_run_config)
            json.dump(run_config, open(run_save_path, 'w'), indent=4)
            print('Run configs dump to %s' % run_save_path)

        try:
            net_save_path = os.path.join(self.path, 'net.config')
            net_config = self.network.config
            if extra_net_config is not None:
                net_config.update(extra_net_config)
            json.dump(net_config, open(net_save_path, 'w'), indent=4)
            print('Network configs dump to %s' % net_save_path)
        except Exception:
            print('%s do not support net config' % type(self.network))

    """ metric related """

    def get_metric_dict(self):
        return {
            'top1': AverageMeter(),
            'top5': AverageMeter(),
        }

    def update_metric(self, metric_dict, output, labels):
        acc1, acc5 = accuracy(output, labels, topk=(1, 5))
        metric_dict['top1'].update(acc1[0].item(), output.size(0))
        metric_dict['top5'].update(acc5[0].item(), output.size(0))

    def get_metric_vals(self, metric_dict, return_dict=False):
        if return_dict:
            return {
                key: metric_dict[key].avg for key in metric_dict
            }
        else:
            return [metric_dict[key].avg for key in metric_dict]

    def get_metric_names(self):
        return 'top1', 'top5'

    """ train and test """

    def validate(self, epoch=0, is_test=False, run_str='', net=None, data_loader=None, no_logs=False, train_mode=False):
        if net is None:
            net = self.net
        # if not isinstance(net, nn.DataParallel):
        #     net = nn.DataParallel(net)

        if data_loader is None:
            data_loader = self.run_config.test_loader if is_test else self.run_config.valid_loader

        if train_mode:
            net.train()
        else:
            net.eval()

        losses = AverageMeter()
        metric_dict = self.get_metric_dict()

        with torch.no_grad():
            para_loader = pl.ParallelLoader(data_loader, [self.device])
            para_loader = para_loader.per_device_loader(self.device)
            with tqdm(total=len(para_loader),
                      desc='Validate Epoch #{} {}'.format(epoch + 1, run_str), disable=no_logs) as t:
                for i, (images, labels) in enumerate(para_loader):
                    images, labels = images.to(self.device), labels.to(self.device)
                    # compute output
                    output = net(images)
                    loss = self.test_criterion(output, labels)
                    # measure accuracy and record loss
                    self.update_metric(metric_dict, output, labels)

                    losses.update(loss.item(), images.size(0))
                    t.set_postfix({
                        'loss': losses.avg,
                        **self.get_metric_vals(metric_dict, return_dict=True),
                        'img_size': images.size(2),
                    })
                    t.update(1)
        return losses.avg, self.get_metric_vals(metric_dict)

    def validate_all_resolution(self, epoch=0, is_test=False, net=None):
        if net is None:
            net = self.network
        if isinstance(self.run_config.data_provider.image_size, list):
            img_size_list, loss_list, top1_list, top5_list = [], [], [], []
            for img_size in self.run_config.data_provider.image_size:
                img_size_list.append(img_size)
                self.run_config.data_provider.assign_active_img_size(img_size)
                self.reset_running_statistics(net=net)
                loss, (top1, top5) = self.validate(epoch, is_test, net=net)
                loss_list.append(loss)
                top1_list.append(top1)
                top5_list.append(top5)
            return img_size_list, loss_list, top1_list, top5_list
        else:
            loss, (top1, top5) = self.validate(epoch, is_test, net=net)
            return [self.run_config.data_provider.active_img_size], [loss], [top1], [top5]

    def train_one_epoch(self, args, epoch, warmup_epochs=0, warmup_lr=0):
        # switch to train mode
        self.net.train()
        MyRandomResizedCrop.EPOCH = epoch  # required by elastic resolution

        nBatch = len(self.run_config.train_loader)

        losses = AverageMeter()
        metric_dict = self.get_metric_dict()
        data_time = AverageMeter()

        with tqdm(total=nBatch,
                  desc='{} Train Epoch #{}'.format(self.run_config.dataset, epoch + 1)) as t:
            end = time.time()
            para_loader = pl.ParallelLoader(self.run_config.train_loader, [self.device])
            para_loader = para_loader.per_device_loader(self.device)
            for i, (images, labels) in enumerate(para_loader):
                MyRandomResizedCrop.BATCH = i
                data_time.update(time.time() - end)
                if epoch < warmup_epochs:
                    new_lr = self.run_config.warmup_adjust_learning_rate(
                        self.optimizer, warmup_epochs * nBatch, nBatch, epoch, i, warmup_lr,
                    )
                else:
                    new_lr = self.run_config.adjust_learning_rate(self.optimizer, epoch - warmup_epochs, i, nBatch)
                new_lr *= xm.xrt_world_size()

                target = labels
                if isinstance(self.run_config.mixup_alpha, float):
                    # transform data
                    lam = random.betavariate(self.run_config.mixup_alpha, self.run_config.mixup_alpha)
                    images = mix_images(images, lam)
                    labels = mix_labels(
                        labels, lam, self.run_config.data_provider.n_classes, self.run_config.label_smoothing
                    )
                images = images.to(self.device)
                labels = labels.to(self.device)

                # compute output
                output = self.net(images)
                loss = self.train_criterion(output, labels)

                # if args.teacher_model is None:
                loss_type = 'ce'

                # compute gradient and do SGD step
                self.net.zero_grad()  # or self.optimizer.zero_grad()
                loss.backward()
                # self.optimizer.step()
                xm.optimizer_step(self.optimizer)

                # measure accuracy and record loss
                losses.update(loss.item(), images.size(0))
                self.update_metric(metric_dict, output, target)

                t.set_postfix({
                    'loss': losses.avg,
                    **self.get_metric_vals(metric_dict, return_dict=True),
                    'img_size': images.size(2),
                    'lr': new_lr,
                    'loss_type': loss_type,
                    'data_time': data_time.avg,
                })
                t.update(1)
                end = time.time()
        return losses.avg, self.get_metric_vals(metric_dict)

    def train(self, args, warmup_epoch=0, warmup_lr=0):

        for epoch in range(self.start_epoch, self.run_config.n_epochs + warmup_epoch):
            train_loss, (train_top1, train_top5) = self.train_one_epoch(args, epoch, warmup_epoch, warmup_lr)

            if (epoch + 1) % self.run_config.validation_frequency == 0:
                img_size, val_loss, val_acc, val_acc5 = self.validate_all_resolution(epoch=epoch, is_test=False)

                is_best = np.mean(val_acc) > self.best_acc
                self.best_acc = max(self.best_acc, np.mean(val_acc))
                val_log = 'Valid [{0}/{1}]\tloss {2:.3f}\t{5} {3:.3f} ({4:.3f})'. \
                    format(epoch + 1 - warmup_epoch, self.run_config.n_epochs,
                           np.mean(val_loss), np.mean(val_acc), self.best_acc, self.get_metric_names()[0])
                val_log += '\t{2} {0:.3f}\tTrain {1} {top1:.3f}\tloss {train_loss:.3f}\t'. \
                    format(np.mean(val_acc5), *self.get_metric_names(), top1=train_top1, train_loss=train_loss)
                for i_s, v_a in zip(img_size, val_acc):
                    val_log += '(%d, %.3f), ' % (i_s, v_a)
                self.write_log(val_log, prefix='valid', should_print=False)
            else:
                is_best = False

            self.save_model({
                'epoch': epoch,
                'best_acc': self.best_acc,
                'optimizer': self.optimizer.state_dict(),
                'state_dict': self.network.state_dict(),
            }, is_best=is_best)

    def reset_running_statistics(self, net=None, subset_size=2000, subset_batch_size=200, data_loader=None):
        from ofa.imagenet_classification.elastic_nn.utils import set_running_statistics
        if net is None:
            net = self.network
        if data_loader is None:
            data_loader = self.run_config.random_sub_train_loader(subset_size, subset_batch_size)
        set_running_statistics(net, data_loader)


def _mp_fn(index, args):
    torch.set_default_tensor_type('torch.FloatTensor')
    distributed_utils.suppress_output(xm.is_master_ordinal())
    main_tpu(args)

def cli_main():
    args = get_args()
    if args.use_gpu:
        return cli_main_gpu(args)
    # From here on out we are in TPU context
    args = adjust_args_tpu(args)
    xmp.spawn(_mp_fn, args=(args,), nprocs=args.num_cores)