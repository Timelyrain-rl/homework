"""
spam和ham表示两类文本，每个文件夹下面分别有25txt文件

请利用朴素贝叶斯文本分类模型完成下面工作：

（1）创建NaiveBayes类

（2）利用NaiveBayes类的train方法训练，选取前20个样本训练

（3）利用NaiveBayes类的predict方法测试，选取5个txt进行测试

（4）计算TP,TN,FP,FN等混淆矩阵的指标
"""
import NaiveBayes_class

naive_bayes = NaiveBayes_class.NaiveBayes()

data, labels = NaiveBayes_class.load_data('email')
X_train, Y_train, X_test, Y_test = NaiveBayes_class.data_split(data, labels)

print(f"训练集文本数量 {len(X_train)}, 类型列表长度 {len(Y_train)},\n验证集文本数量 {len(X_test)}, 类型列表长度 {len(Y_test)}")

naive_bayes.train(X_train, Y_train)

predictions = naive_bayes.predict(X_test)

print("验证集分类结果",predictions)
print("验证集实际分类",Y_test)

# predictions = naive_bayes.predict(X_train)
#
# print("预测集分类结果",predictions)
# print("预测集实际分类",Y_train)

TP, FP, TN, FN = NaiveBayes_class.confusion_matrix(predictions, Y_test, positive='ham', negative='spam')
print(f"\nTrue positive为:{TP}, \nFalse positive为:{FP}, \nTrue negative为:{TN}, \nFalse negative为:{FN}")

# print(naive_bayes.class_word_freq)
# print(naive_bayes.class_total_words)