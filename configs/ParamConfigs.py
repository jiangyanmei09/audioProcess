# -*- coding: UTF-8 -*-
'''
@Project ：audioProcess 
@File    ：ParamConfigs.py
@Author  ：jiangym
@Date    ：2023/4/13 下午8:36
@Description :网络训练参数设置
'''


import argparse


class getParas():
    """"
     This class defines options used during training and testing phase, including
     dataset, feature extractor, network arch, train and test paras.
    """
    def __init__(self):
        self.initialized = False

    def parse_args():
        parser = argparse.ArgumentParser(description="PyTorch Implementation for environmental sound classification")

        # TODO: params for audio feature extractor
        parser.add_argument('--input_size', default=224, type=int, help='image size, like ImageNet:224, cifar:32')


        # TODO: params for datasets
        parser.add_argument('--root_dir', default='./data', type=str, help='the root dir of dataset')
        parser.add_argument('--num_class', default=10, type=int, help='the number of classes, like ImageNet:1000; ESC10: 10')


        # TODO: params for training
        parser.add_argument('--device',      default='cuda',          type=str,   help='whether use cuda to train model or not')
        parser.add_argument('--epoches',    default=10,            type=int,   help='number of total epochs to train')
        parser.add_argument('--bs',        default=32,            type=int,   help='batch size')
        parser.add_argument('--lr',        default=1e-2,          type=float, help='initial learning rate')
        parser.add_argument('--drop',      default=0.5,           type=float, help='dropout ratio')
        parser.add_argument('--weight_decay',   default=1e-4,          type=float, help='weight decay')
        parser.add_argument('--momentum',  default=0.9,           type=float, help='momentum value for SGD optim')
        parser.add_argument('--gama',      default=0.1,           type=float, help='gama update for SGD')
        parser.add_argument('--optimizer', default='SGD',         type=str,   help='optimizer method')
        parser.add_argument('--loss_fn',     default='CrossEntropyLoss', type=str, help='loss function')

        # params for outputs
        parser.add_argument('--log_dir',     default='outputs/logs/',    type=str, help='logs used for tensorboard')
        parser.add_argument('--log_interval', default=100,  type=int, help='训练准确率展示间隔，每隔多少个batch展示一次训练准确率 ')
        parser.add_argument('--save_model_path', default='outputs/models/', type=str, help='save model path')

        return parser.parse_args()

args = getParas.parse_args()
