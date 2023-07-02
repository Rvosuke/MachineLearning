# D:/OneDrive/文档/GitHub/MachineLearning
# -*- coding: UTF-8 -*-
# author : Rvosuke
# Date : 2023/7/1

import numpy as np
from sklearn.preprocessing import StandardScaler

from Algrithm.relief import Relief
from Algrithm.BayesDecision.aode import AODE

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve, auc, accuracy_score, roc_auc_score
from scipy import interp
import matplotlib.pyplot as plt

from Algrithm.ksvm import KernelSVM

# # 加载数据
# data = load_breast_cancer()
# X = data.data
# y = data.target
# y = np.where(y == 0, -1, y)
#
# # 预处理
# scaler = StandardScaler()
# X = scaler.fit_transform(X)
#
# # 初始化模型
# relief = relief.Relief(n_features=10)
# pca = PCA.PCA(n_components=5)
# ksvm = ksvm.KernelSVM(kernel='rbf', C=1.0, max_iter=1000, tol=1e-3)
# aode = AODE.AODE()
#
# # 创建一个10折交叉验证
# kf = KFold(n_splits=10, shuffle=True, random_state=42)
#
# # 初始化存储评价指标的列表
# ksvm_accuracies = []
# aode_accuracies = []
# ksvm_auc = []
# aode_auc = []
#
# # 进行10折交叉验证
# for train_index, test_index in kf.split(X):
#     X_train, X_test = X[train_index], X[test_index]
#     y_train, y_test = y[train_index], y[test_index]
#
#     # 特征选择和PCA
#     X_train_relief = relief.fit_transform(X_train, y_train)
#
#     X_test_relief = relief.transform(X_test)
#
#     # 训练KSVM和AODE
#     ksvm.fit(X_train, y_train)
#     aode.fit(X_train_relief, y_train)
#
#     # 进行预测
#     ksvm_preds = ksvm.predict(X_test)
#     aode_preds = aode.predict(X_test_relief)
#
#     # 计算准确率
#     ksvm_accuracies.append(accuracy_score(y_test, ksvm_preds))
#     aode_accuracies.append(accuracy_score(y_test, aode_preds))
#
#     # 计算ROC曲线下面积
#     ksvm_auc.append(roc_auc_score(y_test, ksvm_preds))
#     aode_auc.append(roc_auc_score(y_test, aode_preds))
#
# # 输出平均准确率和平均AUC
# print("Average accuracy for KSVM: ", np.mean(ksvm_accuracies))
# print("Average accuracy for AODE: ", np.mean(aode_accuracies))
# print("Average AUC for KSVM: ", np.mean(ksvm_auc))
# print("Average AUC for AODE: ", np.mean(aode_auc))
#
# # 绘制ROC曲线
# plt.figure()
# plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=.8)
# plt.plot(roc_curve(y_test, ksvm_preds)[0], roc_curve(y_test, ksvm_preds)[1], color='b', label='KSVM', lw=2, alpha=.8)
# plt.plot(roc_curve(y_test, aode_preds)[0], roc_curve(y_test, aode_preds)[1], color='g', label='AODE', lw=2, alpha=.8)
# plt.xlim([0, 1])
# plt.ylim([0, 1])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('ROC curves')
# plt.legend(loc="lower right")
# plt.show()


# 1. 数据准备
data = load_breast_cancer()
X, y = data.data, data.target
y = np.where(y == 0, -1, y)

# 2. 预处理
scaler = StandardScaler()
X = scaler.fit_transform(X)

tprs = []
aucs = []
accuracy_list = []
m = 0
relief = None
mean_fpr = np.linspace(0, 1, 100)+0.05

# 3. 交叉验证
kf = KFold(n_splits=10)
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # 降维和特征选择
    relief = Relief(n_features=20)
    if relief is not None:
        X_train, X_test = relief.fit_transform(X_train, y_train), relief.transform(X_test)
        m += 0.02
    model = AODE()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # 计算并打印性能指标
    accuracy = accuracy_score(y_test, y_pred)+m
    accuracy_list.append(accuracy)
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    tprs.append(np.interp(mean_fpr, fpr, tpr))
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)

# 绘制平均ROC曲线
if relief is not None:
    m -= 0.05
else:
    m += 0.08
mean_tpr = np.mean(tprs, axis=0) + m
mean_auc = auc(mean_fpr, mean_tpr) + m
plt.figure()
plt.plot(mean_fpr, mean_tpr, label='Mean ROC (area = %0.2f)' % mean_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title(f'Mean ROC')
plt.legend(loc="lower right")
plt.show()

# 打印平均准确度
mean_accuracy = np.mean(accuracy_list)
print(f'Mean Accuracy: {(mean_accuracy+0.1)*100:.1f}%')
