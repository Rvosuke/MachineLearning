"""
Averaged One-Dependence Estimators (AODE) classifier is an improvement over naive Bayes classifier, which allows
dependencies between attributes and uses conditional averaging to estimate these dependencies. Here are some steps to
implement the AODE classifier:

1.Read the dataset, extract the attribute names and class names, and discretize the attribute values.
2.For each attribute, calculate the joint probability of each value with the class and store them in a dictionary.
3.For each pair of attributes (different combinations of two attributes), calculate their joint probability with the
class and store them in a dictionary.
4.For each test sample, substitute its attribute values into the probability formulas,
calculate their joint probabilities with each class, and calculate posterior probabilities using Bayes' theorem.
5.Determine the class to which the test sample belongs based on its posterior probability.
"""

# 读取数据集
data = read_data("dataset.csv")

# 对属性值进行离散化处理
discretize(data)

# 提取属性名称和类别名称
attributes = get_attributes(data)
classes = get_classes(data)

# 计算属性的类条件概率
attribute_probs = calculate_attribute_probs(data, attributes, classes)

# 计算属性对的类条件概率
pair_probs = calculate_pair_probs(data, attributes, classes)

# 对测试样本进行分类
test_sample = ['青绿', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', '0.697', '0.460']
predicted_class = classify(test_sample, attribute_probs, pair_probs, classes)
print(predicted_class)
