# AHNS-in-ReChorus
## sysu ML-class big job
### 项目介绍
自适应难度负样本抽样(AHNS)是一种全新的负样本抽样范式，该方法根据正样本的预测分数动态调整负样本的难度，从而提高协同过滤系统的性能。我们基于轻量级推荐算法框架ReChorus实现AHNS方法。具体AHNS实现位置在`src/general/LightGCN.py`

### 模型运行
1.进入src文件夹
`cd src`
2.执行调用实现AHNS的LightGCN模型
`python .\src\main.py --model_name LightGCN`

### 论文链接
https://arxiv.org/abs/2401.05191
