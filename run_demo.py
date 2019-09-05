import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.datasets.samples_generator import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.base import BaseEstimator, ClassifierMixin


class GBDTLR(BaseEstimator, ClassifierMixin):
    def __init__(self, n_estimators=100, max_depth=3,
                 min_samples_leaf=1, max_leaf_nodes=None,
                 subsample=1.0, learning_rate=0.1,
                 max_iter=100, C=1.0, random_state=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_leaf_nodes = max_leaf_nodes
        self.subsample = subsample
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.C = C
        self.random_state = random_state

        self.gbdt_params = {
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'min_samples_leaf': self.min_samples_leaf,
            'max_leaf_nodes': self.max_leaf_nodes,
            'subsample': self.subsample,
            'learning_rate': self.learning_rate
        }

        self.lr_params = {
            'C': self.C,
            'max_iter': self.max_iter
        }

        self.GBDT = GradientBoostingClassifier(**self.gbdt_params, random_state=random_state)
        self.LR = LogisticRegression(**self.lr_params, random_state=random_state)
        self.ENC = OneHotEncoder(categories='auto')

    def fit(self, X, y):
        X_gbdt, X_lr, Y_gbdt, Y_lr = train_test_split(X, y, test_size=0.5)
        self.GBDT.fit(X_gbdt, Y_gbdt)
        tree_feature = self.GBDT.apply(X_gbdt)[:, :, 0]
        self.ENC.fit(tree_feature)

        X = self.ENC.transform(self.GBDT.apply(X_lr)[:, :, 0])
        y = Y_lr
        return self.LR.fit(X, y)


    def predict(self,X):
        X = self.ENC.transform(self.GBDT.apply(X)[:, :, 0])
        return self.LR.predict(X)


    def predict_proba(self, X):
        X = self.ENC.transform(self.GBDT.apply(X)[:, :, 0])
        return self.LR.predict_proba(X)

    def predict_log_proba(self,X):
        X = self.ENC.transform(self.GBDT.apply(X)[:, :, 0])
        return self.LR.predict_log_proba(X)


if __name__== '__main__':

    # 造点数据
    data_X, data_y = make_classification(n_samples=1000, n_features=30, n_classes=2,
                                         weights=[0.3, 0.7], random_state=1234)
    rng = np.random.RandomState(2)
    data_X += 2 * rng.uniform(size=data_X.shape)
    df = pd.DataFrame(data_X, columns=['fea_' + str(i) for i in range(30)])
    df['label'] = data_y
    # print(df.head())

    # 切分数据集
    X = df.drop(['label'], axis=1)
    Y = df['label']
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.3,random_state=1234)
    # print(X_train.shape, X_test.shape)


    # 测试模型的效果
    params = {
        # 这些是GBDT的超参数
        'n_estimators': 100, 'max_depth': 7,
        'min_samples_leaf': 45, 'max_leaf_nodes': 4,
        'subsample': 0.8, 'learning_rate': 0.1,
        # 这些是LR的超参数
        'max_iter': 2770, 'C': 0.8,
        # random_state是公共参数
        'random_state': 1234
    }

    model = GBDTLR(**params)
    print(model)
    """
    GBDTLR(C=0.8, learning_rate=0.1, max_depth=7, max_iter=2770, max_leaf_nodes=4,
           min_samples_leaf=45, n_estimators=100, random_state=1234, subsample=0.8)
    """

    model.fit(X_train, Y_train)
    Y_pred_proba = model.predict_proba(X_test)
    # print(Y_pred_proba)
    """
    [[7.01777698e-01 2.98222302e-01]
     [4.80790926e-02 9.51920907e-01]
     [1.21897667e-02 9.87810233e-01]
     ...
     [9.73887543e-01 2.61124566e-02]
     [9.39827389e-01 6.01726114e-02]]
    """
    roc_auc = metrics.roc_auc_score(np.array(Y_test), np.array(Y_pred_proba)[:,1])
    print("roc_auc:",roc_auc) #0.93

    Y_pred=model.predict(X_test)
    acc=metrics.accuracy_score(np.array(Y_test),np.array(Y_pred))
    print("acc:",acc) #0.87

    Y_pred_log_proba = model.predict_log_proba(X_test)
    # print(Y_pred_log_proba)

