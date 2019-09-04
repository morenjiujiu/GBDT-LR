# GBDT-LR
GBDT结合LR的二分类模型，封装成了一个类，和sklearn的使用形式一样，有run_demo

## 运行环境
- python3
- numpy, pandas, sklearn

## 注意几个坑
- 要继承自BaseEstimator，否则会有 get_params 这种错误
- __init__()中 “self.n_estimators=n_estimators” 的那些self不能省略，如果只有 self.gbdt_params 和 self.lr_params 是不行的

## 参考
### GBDT+LR的背景
- http://www.cbdio.com/BigData/2015-08/27/content_3750170.htm GBDT+LR的几个迷思。也可以去找facebook的论文来看
- https://scikit-learn.org/stable/auto_examples/ensemble/plot_feature_transformation.html#example-ensemble-plot-feature-transformation-py sklearn官方给出的GBDT+LR的demo
- https://www.zhihu.com/question/329131851/answer/727102009 GBDT+LR一定是好用的吗，缺点
### 创建自定义的estimator，sklearn风格
- https://scikit-learn.org/stable/developers/contributing.html#rolling-your-own-estimator  sklearn官方给出的your-own-estimator说明
- http://danielhnyk.cz/creating-your-own-estimator-scikit-learn/ 国外的一篇博客
