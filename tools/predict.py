# -*- coding: UTF-8 -*-
'''
@Project ：audioProcess 
@File    ：predict.py
@Author  ：jiangym
@Date    ：2023/4/13 下午8:37
@Description :
'''
import os.path
import time
import numpy as np
import torch
from sklearn.metrics import confusion_matrix


from utils.utils import plot_confusion_matrix


class Predictor(object):
    def __init__(self, args, model_path=None):
        # 读取配置参数
        self.configs = args
        self.predictor = None


        # 获取模型
        if model_path is not None:
            model_path = os.path.join(model_path, 'model.pth')
            model_state_dict = torch.load(model_path)
            self.predictor.load_state_dict(model_state_dict)
            print(f'成功加载模型参数：{model_path}')
            self.predictor.eval()
        else:
            self.predictor = self.model

    def predict_batch(self, test_loader, save_matrix_path=None):
            accuracies, preds, labels = [], [], []
            with torch.no_grad():
                for batch_id, (spec_img, label) in enumerate(test_loader):
                    spec_img = spec_img.to(self.configs.device)
                    label = label.to(self.configs.device)
                    output = self.predictor(spec_img)

                    # 模型预测标签
                    pred = np.argmax(output, axis=1)
                    preds.extend(pred.tolist())

                    # 真实标签
                    labels.extend(label.tolist())

                    # 计算one batch size的准确率
                    acc = np.mean((pred == label).astype(int))
                    accuracies.append(acc)
                # 计算整体的准确率
                acc = float(sum(accuracies) / len(accuracies))
                print('Test accuracy is {}'.format(acc))

                # 保存混淆矩阵
                if save_matrix_path is not None:
                    cm = confusion_matrix(labels, preds)
                    plot_confusion_matrix(cm=cm, classes=labels, save_path=os.path.join(save_matrix_path, f'{int(time.time())}.png'))







