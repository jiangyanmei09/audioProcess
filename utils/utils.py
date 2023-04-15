# -*- coding: UTF-8 -*-
'''
@Project ：audioProcess 
@File    ：utils.py
@Author  ：jiangym
@Date    ：2023/4/13 下午8:30
@Description :
'''
import itertools
import os

import matplotlib.pyplot as plt
import numpy as np


def plot_confusion_matrix(cm, save_path, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues, show=False):

    """
    绘制混淆矩阵
    :param cm: 计算出的混淆矩阵
    :param classes: 混淆矩阵中的每一行每一列对应的标签
    :param normalize: True显示百分数，False显示个数
    :param title:
    :param cmap:
    :return:
    """

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("显示百分比")
        np.set_printoptions(formatter={'float': '{: 0.2f}'.format()})
        print(cm)

    else:
        print("显示个数")
        print(cm)

    plt.figure(dpi=300, figsize=(16, 16))
    plt.show(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    # matplotlib版本问题，如果不加下面这行代码，绘制的混淆矩阵只能显示一半，有的版本的matplotlib则不需要加
    plt.ylim(len(classes) - 0.5, -0.5)
    fmt = '.2f' if normalize else 'd'
    # fmt = '.2f'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment='center', color='white' if cm[i, j] > thresh else 'black')

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    # 保存图片
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, format='png')
    if show:
        plt.show()





















