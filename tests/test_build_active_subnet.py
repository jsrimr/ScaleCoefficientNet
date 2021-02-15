import math
import os

import torch
from torchvision import transforms, datasets

from ofa.model_zoo import ofa_net
from ofa.tutorial import evaluate_ofa_subnet


def get_data_loader(imagenet_data_path):
    def build_val_transform(size):
        return transforms.Compose([
            transforms.Resize(int(math.ceil(size / 0.875))),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

    data_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(
            root=os.path.join(imagenet_data_path, 'val'),
            transform=build_val_transform(224)
        ),
        batch_size=250,  # test batch size
        shuffle=True,
        num_workers=16,  # number of workers for the data loader
        pin_memory=True,
        drop_last=False,
    )

    return data_loader


import random
import time
from ofa.tutorial.evolution_finder import ArchManager

import pickle


def test_active_subnet_with_config_and_save_subnet_result():
    # config 에 따라 top1 acc 를 뽑을 수 있음
    # N_network_try = 100

    # exmple net_config
    net_config = {'wid': None,
                  'ks': [7, 5, 5, 3, 7, 5, 3, 5, 7, 5, 3, 7, 5, 7, 7, 3, 5, 3, 7, 5],
                  'e': [4, 4, 6, 3, 3, 4, 4, 4, 6, 6, 4, 3, 6, 3, 3, 3, 6, 6, 6, 3],
                  'd': [2, 3, 3, 4, 4],
                  'r': [208]}

    # results = []
    arch_manager = ArchManager()

    with torch.no_grad():
        # for frame in range(N_network_try):
        ofa_network = ofa_net('ofa_mbv3_d234_e346_k357_w1.2', pretrained=True)
        torch.cuda.empty_cache()
        net_config['r'] = [random.choice(arch_manager.resolutions)]

        for i in range(len(net_config['e'])):
            net_config['ks'][i] = random.choice(arch_manager.kernel_sizes)
            net_config['e'][i] = random.choice(arch_manager.expand_ratios)

        for i in range(len(net_config['d'])):
            net_config['d'][i] = random.choice(arch_manager.depths)

        ofa_network.set_active_subnet(ks=net_config['ks'], d=net_config['d'], e=net_config['e'])

        imagenet_path = "../once-for-all/tutorial/.imagenet_fake/"
        data_loader = get_data_loader(imagenet_path)
        # print(ofa_network.module_str)

        start = time.time()
        top1 = evaluAate_ofa_subnet(
            ofa_network,
            imagenet_path,
            net_config,
            data_loader,
            batch_size=250,
            device='cuda' if torch.cuda.is_available() else 'cpu')
        # device = f'cuda:{random.choice([0, 1, 2, 3])}' if torch.cuda.is_available() else 'cpu')
        latency = time.time() - start

        # results.append({**net_config, 'acc': top1, 'latency': latency})
        # del ofa_network

        # save
        with open(f'../results/acc_{top1}_latency_{latency}.pickle', 'wb') as f:
            pickle.dump({**net_config, 'acc': top1, 'latency': latency}, f, pickle.HIGHEST_PROTOCOL)

        # df = pd.DataFrame(results)
        # df.to_csv("sample_results.csv")

    # evaluate_ofa_subnet 가 결국 아래 로직

    # ofa_network = ofa_network.get_active_subnet().to('cuda:0')
    # calib_bn(ofa_network, imagenet_path, net_config['r'][0], 250) # 이거 없으면 acc 작살난다
    # top1 = validate(ofa_network, imagenet_path, 224, data_loader, 250, 'cuda:0')


if __name__ == '__main__':
    test_active_subnet_with_config_and_save_subnet_result()
    # procs = []
    # for frame in range(N_network_try):
    #     p = Process(target=test_active_subnet_with_config_and_save_subnet_result, args=(frame,))
    #     p.start()
    #     procs.append(p)
    #
    # for p in procs:
    #     p.join()
