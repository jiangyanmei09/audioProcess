# -*- coding: UTF-8 -*-
'''
@Project ：audioProcess 
@File    ：demo.py
@Author  ：jiangym
@Date    ：2023/4/16 下午4:03
@Description : 训练评估模型主函数
'''
import torchvision.datasets

from tools.Trainer import Train
from configs.ParamConfigs import getParas


from torch.utils.data import DataLoader



args = getParas.parse_args()
print(args)


# 构建训练数据和测试数据
# train_dataset = datasets.MyDataset(self.configs.root_dir, self.configs.train_path)
# train_loader = DataLoader(datasets=train_dataset, batch_size=self.configs.batch_size, shuffle=True)
# print("train numbers:{}".format(len(train_dataset)))
#
# test_dataset = datasets.MyDataset(self.configs.root_dir)
# test_loader = DataLoader(datasets=test_dataset, batch_size=batch_size, shuffle=True)
# print("test numbers:{}".format(len(test_dataset)))


train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, transform=torchvision.transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)

test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, transform=torchvision.transforms.ToTensor(), download=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=True)


trainer = Train(args=args)

trainer.train_and_val(resume_model=None,train_loader=train_loader, test_loader=test_loader)

