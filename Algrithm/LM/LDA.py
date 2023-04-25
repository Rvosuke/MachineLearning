import numpy as np

"""
在进行线性判别分析之前，一定需要做数据清洗来把同类别的数据组合起来吗？
"""


class LinearDiscriminantAnalysis:
    def __init__(self, dataset, labels):
        self.dataset = dataset
        self.labels = labels
        # 我刚才在想的一个问题是，如何把一个混乱的数据集按照类别分组
        # 处理方式可以是单独进行一次数据清洗，将同类别的聚集到一起，分成多个文件
        self.classes = np.unique(labels)
        self.mean_vectors = None
        self.cov_matrices = None
        self.within_scatter_matrix = None
        self.weights = None

    def mean_vector(self):
        mean_vectors = []  # 这是存储每个类别均值向量的列表
        for cls in self.classes:
            mean_vectors.append(
                self.dataset[self.labels == cls].mean(axis=0)
            )  # 不同的axis
        self.mean_vectors = np.array(mean_vectors)
        return self.mean_vectors

    def cov_matrix(self):
        cov_matrices = []
        for cls, mean_vec in zip(self.classes, self.mean_vectors):
            class_data = self.dataset[self.labels == cls]
            centered_data = class_data - mean_vec
            cov_matrices.append(np.cov(centered_data.T, bias=False))
        self.cov_matrices = np.array(cov_matrices)
        return self.cov_matrices

    def within_scatter(self):
        self.within_scatter_matrix = np.sum(self.cov_matrices, axis=0)
        return self.within_scatter_matrix

    def weights(self):
        self.weights = np.linalg.inv(self.within_scatter_matrix).dot(self.mean_vectors[1] - self.mean_vectors[0])
        return self.weights

    def fit(self):
        self.mean_vector()
        self.cov_matrix()
        self.within_scatter()
        self.weights()
        return self.weights
