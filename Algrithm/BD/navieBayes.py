"""Watermelon dataset 3.0.Using naive Bayes for classification, assuming that the discrete class conditional
probabilities follow a multinomial (binomial) distribution, and the continuous likelihood follows a normal
distribution. """
import numpy as np
import re

from collections import Counter
from scipy import stats

# 打开文档，将txt文档读取为行向量，去除编号一列
with open('data_3.0.txt', mode='r', encoding='utf-8') as f:
    # 数据集的类别标签.首先分割类别字符串，得到属性的名称
    attributes_labels = next(f).strip().split(',')[1:]
    text = [line.strip().split(',')[1:] for line in f]
text_len = len(text)
attributes_len = len(attributes_labels)

# 导入测试数据，这里直接建立一个测试样本向量
test = ['青绿', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', '0.697', '0.460', '?']


def is_float(s):
    """判断字符串s是否为浮点数"""
    pattern = re.compile(r'^[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?$')
    return bool(pattern.match(s))


# 记录类别标记，并计算先验概率，存储类别标记名称及先验概率，分别存储为行向量
class_count = Counter()
for i in range(len(text)):
    val = text[i][-1]
    class_count[val] += 1
class_len = len(class_count)
prior_p = np.zeros(class_len, dtype=float)

prior_p[:] = [num / text_len for num in class_count.values()]  # 存储先验概率
class_name = list(class_count.keys())  # 存储类别名称

# 计算条件概率，这里利用MLE，之后会更新贝叶斯估计


# 初始化计数列表，对每个离散型样本属性在样本坍塌下进行计数，使用的是Counter
attributes_count_list = []
for i in range(class_len):
    attributes_count = [Counter() for j in range(attributes_len)]
    attributes_count_list.append(attributes_count)
guess_num = 0  # 记录连续型变量的个数
guess_index = set()  # 记录连续变量的序号集合

# 离散型样本属性计数
for row in text:
    for j in range(class_len):
        guess_num = 0
        for i, val in enumerate(row):
            if not is_float(val) and row[-1] == class_name[j] and not i >= attributes_len - 1:
                attributes_count_list[j][i][val] += 1
            if is_float(val):
                guess_num += 1
                guess_index.add(i)

# 根据连续型变量个数构建矩阵来记录，对连续型假设为正态分布
guess_means = np.zeros((class_len, guess_num), dtype=float)
guess_stds = np.zeros((class_len, guess_num), dtype=float)
# 也可以假定为均匀分布，按照MLE，只需要计算出样本中最大值与最小值作为参数估计
uniform_max = np.zeros((class_len, guess_num), dtype=float)
uniform_min = np.zeros((class_len, guess_num), dtype=float)
for i in range(class_len):
    guess_vector = np.zeros((guess_num, class_count[class_name[i]]), dtype=float)
    count = 0
    for row in text:
        if row[-1] == class_name[i]:
            guess_vector[:, count] = [val for val in row if is_float(val)]
            count += 1
    guess_means[i, :] = np.mean(guess_vector, axis=1)  # 计算每行的均值
    guess_stds[i, :] = np.std(guess_vector, axis=1)  # 计算每行的标准差
    uniform_max[i, :] = np.max(guess_vector, axis=1)  # 计算每行的最大值
    uniform_min[i, :] = np.min(guess_vector, axis=1)  # 计算每行的最小值

# 接下来计算样本属性的各个类条件概率
condition_p = np.zeros((class_len, attributes_len), dtype=float)
for k in range(class_len):
    for i, val in enumerate(test[:-1]):
        if not is_float(val):
            class_condition_p = [attributes_count_list[k][i][val] / class_count[class_name[k]]
                                 for k in range(class_len)]
        else:
            j = list(guess_index).index(i)
            class_condition_p = [stats.norm.pdf(float(val), guess_means[k][j], guess_stds[k][j])
                                 for k in range(class_len)]
            # class_condition_p = [1 / (uniform_max[k][j] + uniform_min[k][j]) for k in range(class_len)]  # 均匀分布估计
        condition_p[:, i][:] = class_condition_p

# 接着进行判别函数的计算，这里是朴素贝叶斯，条件独立性假设
class_g = np.zeros(class_len, dtype=float)
for i in range(class_len):
    class_g[i] = prior_p[i] * np.prod(condition_p[i])

# 根据判别函数做出决策
decision = class_name[np.argmax(class_g)]
print(decision)
exit(0)
