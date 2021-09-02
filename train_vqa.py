from __future__ import absolute_import, division, print_function
import os

import torch.optim
from data_loader._dataset_ import _dataset_, Dataset_this
from torch.utils.data import DataLoader
from config_parser import config_parser
from model.FR_model import fr_VQA_model
from scipy.io import savemat

# import pickle
import numpy as np
import random


def train_vqa(config_file=None, output_path=None, is_random=False, target=None):

    # repeatable
    if not is_random:
        seed_torch(12318)

    # load config data
    [db_config, model_config] = config_parser(config_file=config_file)

    # load data
    dataset_init = _dataset_(db_config, target=target)
    dataset_train = Dataset_this(dataset_init.train_dict, is_train=True, is_shuffle=False)
    dataset_test = Dataset_this(dataset_init.test_dict, is_train=False, is_shuffle=False)
    del dataset_init

    #
    train_batch_size = int(model_config.get('train_batch_size', 2))
    train_loader = DataLoader(dataset_train, batch_size=train_batch_size, shuffle=False, drop_last=True)
    del dataset_train

    test_batch_size = int(model_config.get('test_batch_size', 2))
    test_loader = DataLoader(dataset_test, batch_size=test_batch_size, shuffle=False, drop_last=False)
    del dataset_test

    # initialize model
    model = fr_VQA_model(model_config)
    print('number of trainable parameters = ', count_parameters(model.model))

    # set optimizer
    optimizer = torch.optim.Adam(model.model.parameters(), model.lr, betas=(0.9, 0.999))
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)  # update lr schedule

    # weight_p, bias_p = [], []
    # for name, p in model.model.named_parameters():
    #     if 'bias' in name:
    #         bias_p += [p]
    #     else:
    #         weight_p += [p]
    # optimizer = torch.optim.Adam([{'params': weight_p, 'weight_decay': 1e-5},
    #                               {'params': bias_p, 'weight_decay': 0}],
    #                              model.lr, betas=(0.9, 0.999))

    # run
    best_srcc, best_epoch, info = \
        model.run(train_loader, test_loader, optimizer)
    # model.eval(test_loader)

    savemat(output_path + '/%s.mat' % ('_'.join(target.astype(str))), {'info': info})
    # with open(output_path + '/%s.pkl' % ('_'.join(target.astype(str))), 'wb') as f:
    #     pickle.dump([loss_train, loss_eval, srcc_eval], f)

    return best_srcc, best_epoch


def seed_torch(seed=12318):

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print('--' * 12)
    print('seed: %d' % seed)


def count_parameters(model):
    total_param = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            num_param = np.prod(param.size())
            if param.dim() > 1:
                print(name, ': ', 'x'.join(str(x) for x in list(param.size())), '=', num_param)
            else:
                print(name, ': ', num_param)
            total_param += num_param
    return total_param
