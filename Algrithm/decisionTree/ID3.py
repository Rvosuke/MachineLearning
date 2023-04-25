"""
下边我要实现一个ID3算法，用于决策树的生成
输入一个数据集，输出一个决策树
首先进行一步判断，判断数据集中的样本是否属于同一类别
如果是，返回该类别
如果不是，进行下一步判断
判断数据集中的特征是否相同
如果是，返回样本中数量最多的类别
如果不是，进行下一步判断
判断当前数据集中的样本是否为空
如果是，返回其父节点中数量最多的类别
如果不是，进行下一步判断
计算数据集中每个特征的信息增益，选择信息增益最大的特征，这一步以一个函数来实现
根据该特征的不同取值，将数据集划分为若干个子集
对每个子集，递归调用上述步骤，得到一个子树
将所有子树组合成一个决策树
"""
import numpy as np
import pandas as pd
from collections import Counter


def calc_entropy(y):
    """计算信息熵"""
    counts = np.bincount(y)  # 统计y中每个值出现的次数
    probs = counts / len(y)  # 计算每个值出现的概率
    entropy = -np.sum([p * np.log2(p) for p in probs if p > 0])  # 计算信息熵
    print(f"Entropy calculation: -Σ(p * log2(p)) for {dict(Counter(y))}")
    print(f"Entropy: {entropy}\n")
    return entropy


def calc_info_gain(DATA, feature, TARGET):
    """计算信息增益"""
    original_entropy = calc_entropy(DATA[TARGET])  # 计算原始信息熵
    feature_values = DATA[feature].unique()  # 计算特征的取值
    weighted_entropy = 0  # 计算加权后的信息熵
    print(f"Information gain calculation for feature '{feature}':")

    for value in feature_values:  # 遍历每个特征取值
        subset = DATA[DATA[feature] == value]  # 划分子集
        entropy = calc_entropy(subset[TARGET])  # 计算子集的信息熵
        weighted_entropy += (len(subset) / len(DATA)) * entropy  # 计算加权后的信息熵

    info_gain = original_entropy - weighted_entropy  # 计算信息增益
    print(f"Information gain for feature '{feature}': {info_gain}\n")
    return info_gain


def majority_class(y):
    """返回样本中数量最多的类别"""
    return y.value_counts().idxmax()


def id3(DATA, FEATURES, TARGET, parent_data=None):
    """ID3算法"""
    if len(DATA) == 0:
        """如果数据集为空，返回其父节点中数量最多的类别"""
        return majority_class(parent_data[TARGET])

    if len(DATA[TARGET].unique()) == 1:
        """如果数据集中的样本属于同一类别，返回该类别"""
        return DATA[TARGET].iloc[0]

    if len(FEATURES) == 0:
        """如果数据集中的特征为空，返回其父节点中数量最多的类别"""
        return majority_class(DATA[TARGET])

    info_gains = {feature: calc_info_gain(DATA, feature, TARGET) for feature in FEATURES}  # 计算每个特征的信息增益
    best_feature = max(info_gains, key=info_gains.get)  # 选择信息增益最大的特征

    tree = {best_feature: {}}  # 创建决策树
    remaining_features = [f for f in FEATURES if f != best_feature]  # 去掉最优特征

    for value in DATA[best_feature].unique():  # 遍历最优特征的每个取值
        subset = DATA[DATA[best_feature] == value]  # 划分子集
        tree[best_feature][value] = id3(subset, remaining_features, TARGET, DATA)  # 递归调用ID3算法

    return tree


# Example usage:
data = pd.DataFrame({
    'x1': [1, 1, 1, 1, 2, 2, 2, 2],
    'x2': [1, 1, 2, 2, 1, 1, 2, 2],
    'x3': [1, 2, 1, 2, 1, 2, 1, 2],
    'y': [3, 1, 3, 2, 3, 2, 3, 3]
})

features = ['x1', 'x2', 'x3']
target = 'y'

decision_tree = id3(data, features, target)
print(decision_tree)
