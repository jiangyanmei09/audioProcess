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


#TODO:混淆矩阵中classes是怎么与cm对应的，classes的类型是怎样的
def plot_confusion_matrix(cm, save_path, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues, show=False):

    """
    绘制混淆矩阵
    :param cm: 计算出的混淆矩阵
    :param classes: 混淆矩阵中的每一行每一列对应的标签,不是一个数值，而是数据的标签，比如CIFAR10就是对应的10个标签
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
    # m = np.max(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
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


# def plot_confusion_matrix(cm, save_path, class_labels, title='Confusion Matrix', show=False):
#     plt.figure(figsize=(12, 8), dpi=100)
#     np.set_printoptions(precision=2)
#     # 在混淆矩阵中每格的概率值
#     ind_array = np.arange(len(class_labels))
#     x, y = np.meshgrid(ind_array, ind_array)
#     for x_val, y_val in zip(x.flatten(), y.flatten()):
#         c = cm[y_val][x_val] / (np.sum(cm[:, x_val]) + 1e-6)
#         # 忽略值太小的
#         if c < 1e-4: continue
#         plt.text(x_val, y_val, "%0.2f" % (c,), color='red', fontsize=15, va='center', ha='center')
#     m = np.max(cm)
#     plt.imshow(cm / m, interpolation='nearest', cmap=plt.cm.binary)
#     plt.title(title)
#     plt.colorbar()
#     xlocations = np.array(range(len(class_labels)))
#     plt.xticks(xlocations, class_labels, rotation=90)
#     plt.yticks(xlocations, class_labels)
#     plt.ylabel('Actual label')
#     plt.xlabel('Predict label')
#
#     # offset the tick
#     tick_marks = np.array(range(len(class_labels))) + 0.5
#     plt.gca().set_xticks(tick_marks, minor=True)
#     plt.gca().set_yticks(tick_marks, minor=True)
#     plt.gca().xaxis.set_ticks_position('none')
#     plt.gca().yaxis.set_ticks_position('none')
#     plt.grid(True, which='minor', linestyle='-')
#     plt.gcf().subplots_adjust(bottom=0.15)
#     # 保存图片
#     os.makedirs(os.path.dirname(save_path), exist_ok=True)
#     plt.savefig(save_path, format='png')
#     if show:
#         # 显示图片
#         plt.show()




















