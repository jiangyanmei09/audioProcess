# audioProcess
基于pytorch的声音分类，包括数据的读取、特征提取、模型搭建、训练、评价。


# 项目结构

- configs:配置参数
- data：数据
- data_utils：数据读取、生成数据集、加载数据集
- extractor：特征提取，数据增强
- models：模型结构
- outputs：输出的结果。
  - logs：日志
  - checkpoints:保存的模型
  - images:可视化的结果图片
- tools：包括训练，评估和预测
  - train.py
  - eval.py
  - predict.py
- utils:辅助函数，可以是评价指标，调用的其他函数等