# -*- coding: UTF-8 -*-
'''
@Project ：audioProcess 
@File    ：Trainer.py
@Author  ：jiangym
@Date    ：2023/4/13 下午8:31
@Description : 类名使用大驼峰命名法，eg.MyDataset,ParamConfigs
            方法名，函数名，成员变量，局部变量使用小驼峰命名法，eg.getParams, setupModel
'''
import os.path
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
from tensorboardX import tensorbord

import models
from configs.ParamConfigs import getParas
from data_utils import datasets
from configs import ParamConfigs
from utils.utils import plot_confusion_matrix

args = getParas.parse_args()

class Train(object):
    def __init__(self, args):

        """
        训练、评估模型集成工具
        :param configs: 配置字典
        """

        # 读取配置参数
        self.configs = args
        self.configs.device = args.cuda
        self.configs.epoches = args.epoches
        self.configs.optimizer = args.optimizer
        self.configs.loss_fn = args.loss_fn
        self.configs.batch_size = args.batch_size
        self.configs.lr = args.lr
        self.configs.weight_decay = args.weight_decay

        self.model = None
        self.test_loader = None
        self.tensorboard = tensorboard.SummarWriter(self.configs.log_dir) # tensorboard保存目录


    def __setupModel(self):
        """
        模型架构、优化器、损失函数的选择
        :param args:
        :return:
        """
        # 获取模型
        # TODO：后面增加if函数，用于选择不同的模型架构，在网络架构的最后一层，需要考虑类别数量，才能设置fcn层输出神经元个数

        self.model = models.ACRNN()

        # 获取优化器
        if self.configs.optimizer == 'Adam':
            self.optimizer = torch.optim.Adam(params=self.model.parameters(),
                                              lr=self.configs.lr,
                                              weight_decay=self.configs.weight_decay)

        elif self.configs.optimizer == 'SGD':
            self.optimizer = torch.optim.SGD(params=self.model.parameters(),
                                        momentum=ParamConfigs.args.momentum,
                                        lr=self.configs.lr,
                                        weight_decay=self.configs.weight_decay)

        elif self.configs.optimizer == 'AdamW':
            self.optimizer = torch.optim.AdamW(params=self.model.parameters(),
                                          lr=self.configs.lr,
                                          weight_decay=self.configs.weight_decay)

        else:
            raise Exception(f'不支持的优化方法：{self.configs.optimizer}')

        # 获取损失函数
        if self.configs.loss_fn == 'CrossEntropyLoss':
            self.loss_fn = torch.nn.CrossEntropyLoss()

        elif self.configs.loss_fn == 'MSE':
            self.loss_fn = torch.nn.MSELoss()

        else:
            raise Exception(f'不支持的损失函数：{args.loss}')



    def __train_epoch(self, epoch_id):
        """
        training process of one epoch which means one iteration.
        it only uses a mini-batch data to train network and does one gradient descent.
        Note that  mini-batch will be divided before each epoch.
        :param epoch_id:
        :return: accuracy and loss of each epoch
        """

        train_time, accuracies, loss_sum = [], [], []
        start = time.time()
        sum_batch = len(self.train_loader) * ParamConfigs.args.epoch

        for batch_id, (spec_img, label) in enumerate(self.train_loader):
            spec_img = spec_img.to(self.configs.device)
            label = label.to(self.configs.device)
            output = self.model(spec_img)

            # 计算损失函数，反向传播，更新权重参数
            loss = self.loss_fn(output, label)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # 计算准确率---for each batch
            output = torch.nn.functional.softmax(output, dim=-1)
            output = output.data.cpu().numpy()
            output = np.argmax(output, axis=1)
            label = label.data.cpu().numpy()
            acc = np.mean((output == label).astype(int))
            accuracies.append(acc)
            loss_sum.append(loss)
            train_time.append((time.time() - start) * 1000)

            if batch_id % self.configs.log_interval == 0:
                print(f'Train epoch: [{epoch_id}/{ParamConfigs.args.epoch}]')
                print(f'batch: [{batch_id}/{len(self.train_loader)}]')
                print(f'loss: [{sum(loss_sum) / len(loss_sum)}]')
                print(f'accuracy: [{sum(accuracies) / len(accuracies)}]')

        # for each epoch
        accuracy = sum(accuracies) / len(accuracies)
        loss = sum(loss_sum) / len(loss_sum)
        return accuracy, loss


    def train_and_val(self, resume_model=None):

        # 构建训练数据和测试数据
        train_dataset = datasets.MyDataset(self.configs.root_dir, self.configs.train_path)
        train_loader = DataLoader(datasets=train_dataset, batch_size=self.configs.batch_size, shuffle=True)
        print("train numbers:{}".format(len(train_dataset)))

        test_dataset = datasets.MyDataset(self.configs.root_dir, self.configs.test_path)
        test_loader = DataLoader(datasets=test_dataset, batch_size=self.configs.batch_size, shuffle=True)
        print("test numbers:{}".format(len(test_dataset)))

        # 模型架构，优化器，损失函数,在configs.py文件中设置参数
        self.__setupModel()
        self.model.to(self.configs.device)
        self.loss_fn.to(self.configs.device)

        # 开始训练模型，并用验证集评估模型

        for epoch in range(self.configs.epoches):
            start_epoch = time.time()

            # 训练一个epoch
            train_acc, train_loss = self.__train_epoch(epoch_id=epoch)

            # 验证准确率，损失
            val_acc, val_loss = self.evaluate(resume_model=None)

            self.tensorboard.add_scalar("train_acc", train_acc, epoch)
            self.tensorboard.add_scalar("train_loss", train_loss, epoch)
            self.tensorboard.add_scalar("val_acc", val_acc, epoch)
            self.tensorboard.add_scalar("val_loss", val_loss, epoch)

            self.save_model(epoch, val_acc)



    def evaluate(self, resume_model='outputs/checkpoints/', save_matrix_path=None):
        """
        模型路径存在，则加载模型后，用验证集数据进行评估，若不存在的话，则直接使用前期训练的模型进行评估
        :param resume_model: 所使用的模型
        :param save_matrix_path: 保存混淆矩阵的路径
        :return: 评估结果
        """

        # 加载模型
        if resume_model is not None:
            if os.path.isdir(resume_model):
                resume_model = os.path.join(resume_model, 'model.pt')
                assert os.path.exists(resume_model), f"{resume_model} 模型不存在！"
                model_state_dict = torch.load(resume_model)
                self.model.load_state_dict(model_state_dict)
                print(f"成功加载模型：{resume_model}")
            self.model.eval()

        if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            eval_model = self.model.module()
        else:
            eval_model = self.model

        accuracies, loss_sum, preds, labels = [], [], [], []
        with torch.no_grad():
            for batch_id, (spec_img, label) in enumerate(self.test_loader):
                spec_img = spec_img.to(self.configs.device)
                label = label.to(self.configs.device)
                output = eval_model(spec_img)

                # 计算损失
                loss = self.loss_fn(output, label)

                label = label.data.cpu().numpy()
                output = output.data.cpu().numpy()

                # 模型预测标签
                pred = np.argmax(output, axis=1)
                preds.extend(pred.tolist())

                # 真实标签
                labels.extend(label.tolist())

                # 计算one batch size 的准确率
                acc = np.mean((pred == label).astype(int))
                accuracies.append(acc)
                loss_sum.append(loss.data().cpu().numpy())

        # 计算整体的准确率和损失
        loss = float(sum(loss_sum) / len(loss_sum))
        acc = float(sum(accuracies) / len(accuracies))

        # 保存混淆矩阵
        if save_matrix_path is not None:
            cm = confusion_matrix(labels, preds)
            plot_confusion_matrix(cm=cm, save_matrix_path=os.path.join(save_matrix_path, f'{int(time.time())}.png'),
                                  classes=self.configs.num_class)

        self.model.train()
        return loss, acc

    def __save_model(self, save_model_path, epoch_id, best_acc):
        model_path = os.path.join(self.configs.model_dir, 'model_{:0=3d}_{:.3f}.pt'.format(epoch_id, best_acc))

        if not os.path.join(os.path.dirname(model_path)):
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.jit.save(torch.jit.script(self.model), model_path)




























