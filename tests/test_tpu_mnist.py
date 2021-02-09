"""
most code from
https://colab.research.google.com/github/pytorch/xla/blob/master/contrib/colab/mnist-training.ipynb#scrollTo=pTmxZL5ymp8P
"""

import math
from matplotlib import pyplot as plt

M, N = 4, 6
RESULT_IMG_PATH = '/tmp/test_result.png'

def plot_results(images, labels, preds):
  images, labels, preds = images[:M*N], labels[:M*N], preds[:M*N]
  inv_norm = transforms.Normalize((-0.1307/0.3081,), (1/0.3081,))

  num_images = images.shape[0]
  fig, axes = plt.subplots(M, N, figsize=(11, 9))
  fig.suptitle('Correct / Predicted Labels (Red text for incorrect ones)')

  for i, ax in enumerate(fig.axes):
    ax.axis('off')
    if i >= num_images:
      continue
    img, label, prediction = images[i], labels[i], preds[i]
    img = inv_norm(img)
    img = img.squeeze() # [1,Y,X] -> [Y,X]
    label, prediction = label.item(), prediction.item()
    if label == prediction:
      ax.set_title(u'\u2713', color='blue', fontsize=22)
    else:
      ax.set_title(
          'X {}/{}'.format(label, prediction), color='red')
    ax.imshow(img)
  plt.savefig(RESULT_IMG_PATH, transparent=True)

# Define Parameters
FLAGS = {}
FLAGS['datadir'] = "/tmp/mnist"
FLAGS['batch_size'] = 128
FLAGS['num_workers'] = 4
FLAGS['learning_rate'] = 0.01
FLAGS['momentum'] = 0.5
FLAGS['num_epochs'] = 10
FLAGS['num_cores'] = 8
FLAGS['log_steps'] = 20
FLAGS['metrics_debug'] = False

import numpy as np
import os
import time
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
from torchvision import datasets, transforms

SERIAL_EXEC = xmp.MpSerialExecutor()


class MNIST(nn.Module):

    def __init__(self):
        super(MNIST, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.bn1 = nn.BatchNorm2d(10)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.bn2 = nn.BatchNorm2d(20)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = self.bn1(x)
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = self.bn2(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


# Only instantiate model weights once in memory.
WRAPPED_MODEL = xmp.MpModelWrapper(MNIST())


def train_mnist():
    torch.manual_seed(1)

    """
    tpu 를 쓴다하면 dataset 에 할 일
    train_dataset, test_dataset = SERIAL_EXEC.run(get_dataset)
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset,
        num_replicas=xm.xrt_world_size(),
        rank=xm.get_ordinal(),
        shuffle=True)
    """
    def get_dataset():
        norm = transforms.Normalize((0.1307,), (0.3081,))
        train_dataset = datasets.MNIST(
            FLAGS['datadir'],
            train=True,
            download=True,
            transform=transforms.Compose(
                [transforms.ToTensor(), norm]))
        test_dataset = datasets.MNIST(
            FLAGS['datadir'],
            train=False,
            download=True,
            transform=transforms.Compose(
                [transforms.ToTensor(), norm]))

        return train_dataset, test_dataset

    # Using the serial executor avoids multiple processes to
    # download the same data.
    train_dataset, test_dataset = SERIAL_EXEC.run(get_dataset)

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset,
        num_replicas=xm.xrt_world_size(),
        rank=xm.get_ordinal(),
        shuffle=True)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=FLAGS['batch_size'],
        sampler=train_sampler,
        num_workers=FLAGS['num_workers'],
        drop_last=True)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=FLAGS['batch_size'],
        shuffle=False,
        num_workers=FLAGS['num_workers'],
        drop_last=True)

    # Scale learning rate to world size
    lr = FLAGS['learning_rate'] * xm.xrt_world_size()

    # Get loss function, optimizer, and model
    """
    tpu 쓴다하면 device 가 
    device = xm.xla_device()
    model = xmp.MpModelWrapper(MNIST()).to(device)
    """
    device = xm.xla_device()
    model = WRAPPED_MODEL.to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=FLAGS['momentum'])
    loss_fn = nn.NLLLoss()

    def train_loop_fn(loader):
        tracker = xm.RateTracker()
        model.train()
        for x, (data, target) in enumerate(loader):
            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, target)
            loss.backward()
            # tpu 쓴다하면 optimizer 에 xm.optimizer_step(optimizer)
            xm.optimizer_step(optimizer)
            tracker.add(FLAGS['batch_size'])
            if x % FLAGS['log_steps'] == 0:
                print('[xla:{}]({}) Loss={:.5f} Rate={:.2f} GlobalRate={:.2f} Time={}'.format(
                    xm.get_ordinal(), x, loss.item(), tracker.rate(),
                    tracker.global_rate(), time.asctime()), flush=True)

    def test_loop_fn(loader):
        total_samples = 0
        correct = 0
        model.eval()
        data, pred, target = None, None, None
        for data, target in loader:
            output = model(data)
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
            total_samples += data.size()[0]

        accuracy = 100.0 * correct / total_samples
        print('[xla:{}] Accuracy={:.2f}%'.format(
            xm.get_ordinal(), accuracy), flush=True)
        return accuracy, data, pred, target

    # Train and eval loops
    accuracy = 0.0
    data, pred, target = None, None, None
    for epoch in range(1, FLAGS['num_epochs'] + 1):
        para_loader = pl.ParallelLoader(train_loader, [device])
        train_loop_fn(para_loader.per_device_loader(device))
        xm.master_print("Finished training epoch {}".format(epoch))

        para_loader = pl.ParallelLoader(test_loader, [device])
        accuracy, data, pred, target = test_loop_fn(para_loader.per_device_loader(device))
        if FLAGS['metrics_debug']:
            xm.master_print(met.metrics_report(), flush=True)

    return accuracy, data, pred, target

# Start training processes
def _mp_fn(rank, flags):
  global FLAGS
  FLAGS = flags
  torch.set_default_tensor_type('torch.FloatTensor')
  accuracy, data, pred, target = train_mnist()
  if rank == 0:
    # Retrieve tensors that are on TPU core 0 and plot.
    plot_results(data.cpu(), pred.cpu(), target.cpu())


import time
def test_multiple_tpu_mnist():
    start_time = time.time()
    xmp.spawn(_mp_fn, args=(FLAGS,), nprocs=FLAGS['num_cores'],
              start_method='fork')

    print("Time taken : ", time.time() - start_time)

if __name__ == '__main__':
    test_multiple_tpu_mnist()