# AHNS-in-ReChorus
## sysu ML-class work
### 项目介绍
自适应难度负样本抽样(AHNS)是一种全新的负样本抽样范式，该方法根据正样本的预测分数动态调整负样本的难度，从而提高协同过滤系统的性能。我们基于轻量级推荐算法框架ReChorus实现AHNS方法。

### 参考论文链接
https://arxiv.org/abs/2401.05191

### 实现位置
AHNS的具体实现位置在`src/general/LightGCN.py`

### 模型运行
1.进入src文件夹
```
cd src
```
2.执行调用实现AHNS的LightGCN模型
```
python .\src\main.py --model_name LightGCN
```

### 实验结果
在Grocery_and_Gourmet_Food和MovieLens-1M数据集上进行实验，实验结果部分参数如下：

![7164ae587800fe4cda928371f23b448](https://github.com/user-attachments/assets/4544797a-4196-4000-abb4-ea2779e2675a)

![8ce74e5354c08b13d6bba6a9f99d2e6](https://github.com/user-attachments/assets/4169e2c5-68db-4ecc-ab43-600fdb39bd57)
