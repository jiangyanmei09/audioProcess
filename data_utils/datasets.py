# -*- coding: UTF-8 -*-
'''
@Project ：audioProcess 
@File    ：datasets.py
@Author  ：jiangym
@Date    ：2023/4/13 下午8:29
@Description : 数据读取dataset，必须继承torch.utils.data.Dataset，只需要实现__init__构造函数，__getitem__迭代器遍历函数以及__len__函数。
                __init__函数读取传入的数据集路径下的指定数据文件
                __getitem__函数根据传入的索引从data_path中取对应的元素
                __len__返回数据集的大小
'''
import os
import librosa
from torch.utils.data import Dataset
from torchvision import transforms



class MyDataset(Dataset):

    # 获取文件的路径和每个数据的名称, 适用于同一样本放在同一个文件夹下的情况
    def __init__(self, root_dir, label_dir, transforms):
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.transforms = transforms
        self.path = os.path.join(root_dir, label_dir)
        self.data_path = os.listdir(self.path)


    # 获取样本数据和标签
    def __getitem__(self, item):
        data_name = self.data_path[item]
        data_item_path = os.path.join(self.root_dir, self.data_path, data_name)
        audio, sr = librosa.load(data_item_path)
        label = self.label_dir
        return audio, sr, label


    # 获取数据的长度

    def __len__(self):
        return len(self.data_path)



if __name__ == "__main__":
    root_dir = '../data'
    label_dir = '1'
    datasets = MyDataset(root_dir, label_dir)


transforms.Compose([transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406])])



