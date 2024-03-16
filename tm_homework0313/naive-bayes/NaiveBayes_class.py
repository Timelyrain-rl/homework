import math
import os
import re


class NaiveBayes:

    def __init__(self):
        # set of unique words in the training set
        # 词汇表
        # 集合类型
        self.vocab = set()

        # word frequency count for each class
        # 每一类文本的词频统计
        # 字典类型
        self.class_word_freq = {}

        # total word count for each class
        # 每一类文本的总词汇数
        # 字典类型
        self.class_total_words = {}

        # unique classes in the training set
        # 训练集中文本的类型
        # 列表类型
        self.classes = []

    def train(self, X, Y):
        """

        Train the Naive Bayes classifier on the given training data.

        X: list of training instances (list of strings)

        Y: list of training labels (list of strings)

       """

        # 传入训练集文本
        for i in range(len(X)):
            bef_words = X[i].split()
            label = Y[i]
            words = [bef_word.lower() for bef_word in bef_words if re.match('^[a-zA-Z]+$', bef_word)]

            # 将类放入存放类型的列表中，只需要传入新的
            if label not in self.classes:
                self.classes.append(label)
            # 将文本中的词添加到词汇表
            self.vocab.update(words)

            # 创建按类将词频进行存放的字典
            if label not in self.class_word_freq:
                self.class_word_freq[label] = {}

            # 统计词频
            for word in words:
                self.class_word_freq[label][word] = self.class_word_freq[label].get(word, 0) + 1

            # 将本次传入文本的总词汇数添加到该类的总词汇数中
            self.class_total_words[label] = self.class_total_words.get(label, 0) + len(words)

        # 查看训练集类型顺序
        print(f"\n训练集类型的顺序为:{self.classes}\n")

    def predict(self, X):
        """

        Predict the class labels for the given test data.

        X: list of test instances (list of strings)

        Returns: list of predicted labels (list of strings)

        """

        # 创建存放验证集类型的列表
        predictions = []

        # 遍历验证集
        for instance in X:
            words = instance.split()

            # 验证集中文本是某类型的概率
            else_prob = float('-inf')
            # 验证集中文本的类型
            text_class = None

            # 遍历类型列表
            # 分别计算该文本是某类型的概率
            # 概率大的就分类到某类型
            for label in self.classes:
                prob = 0
                for word in words:
                    # 平滑处理
                    word_freq = self.class_word_freq.get(label, {}).get(word, 0) + 1
                    word_prob = word_freq / (self.class_total_words.get(label, 0) + len(self.vocab))
                    # 将概率相加处理
                    prob = prob + math.log(word_prob)

                # 如果在某类型的概率大于在另外一个类型
                # 那么就把文本归类到概率大的一方
                if prob > else_prob:
                    else_prob = prob
                    text_class = label

            # 把预测结果加入到列表中
            predictions.append(text_class)

        return predictions


def load_data(folder_path):
    """
    从email文件夹加载数据集

    :param folder_path: 这里就是email文件夹，文件夹下有两个以类型命名的文件夹
    :return: 返回文件内容列表 data，文件类型列表 labels
    """
    data = []
    labels = []

    # email文件夹下两个文件夹的名字即为分类
    for label in os.listdir(folder_path):
        label_folder = os.path.join(folder_path, label)
        # print(label_folder)
        if os.path.isdir(label_folder):

            for file_name in os.listdir(label_folder):
                file_path = os.path.join(label_folder, file_name)

                if os.path.isfile(file_path):

                    with open(file_path, 'r', encoding='latin-1') as file:
                        content = file.read()
                        data.append(content)
                        labels.append(label)

    return data, labels


def data_split(data, labels):
    """
    数据集划分
    要求的数据集是4：1，按照该比例进行划分

    :param data: load_data 返回的 data 列表
    :param labels: load_data 返回的 labels 列表
    :return: 返回划分好的数据集，分别是训练集的内容(X_train)与标签(Y_train)，验证集的内容(X_test)与标签(Y_test)
    """
    X_train = []
    Y_train = []
    X_test = []
    Y_test = []

    # 遍历类型
    for label in set(labels):
        # 按类别依次划分数据集
        # 数据集中的顺序与NaiveBayes中的类型属性顺序一致
        label_data = [data[i].replace('\n', ' ') for i in range(len(data)) if labels[i] == label]
        label_split_index = int(len(label_data) * 0.8)
        print(f"{label_data}\n")
        X_train.extend(label_data[:label_split_index])
        Y_train.extend([label] * label_split_index)
        X_test.extend(label_data[label_split_index:])
        Y_test.extend([label] * (len(label_data) - label_split_index))

    return X_train, Y_train, X_test, Y_test


def confusion_matrix(labels_predictions, labels_test, positive, negative):
    """
    计算混淆矩阵的值

    :param labels_predictions: NaiveBayes类中返回的预测列表
    :param labels_test: 数据集划分时得到的正确属性
    :param positive: 正例，这里是ham
    :param negative: 反例，这里是spam
    :return: 返回四个混淆矩阵
    """
    TP, FP, TN, FN = 0, 0, 0, 0

    for label_index in range(len(labels_predictions)):
        # TP代表正确分类为正例的数量
        if labels_predictions[label_index] == labels_test[label_index] and labels_predictions[label_index] == positive:
            TP = TP + 1

        # TN代表正确分类为反例的数量
        elif labels_predictions[label_index] == labels_test[label_index] and labels_predictions[label_index] == negative:
            TN = TN + 1

        # FP代表错误分类为正例的数量
        elif labels_predictions[label_index] != labels_test[label_index] and labels_predictions[label_index] == positive:
            FP = FP + 1

        # FN代表误错分类为反例的数量
        elif labels_predictions[label_index] != labels_test[label_index] and labels_predictions[label_index] == negative:
            FN = FN + 1

    return TP, FP, TN, FN
