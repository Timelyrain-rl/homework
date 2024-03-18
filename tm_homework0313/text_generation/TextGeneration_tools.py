import os
import random
import jieba


def stopwords_load(folder_path):
    """
    加载停用词
    这里的停用词主要是标点符号之类的
    像是逗号句号之类的标点符号不需要去除
    去除它们反而会使文本没有结束语句，显得很乱
    极端一点可能所有标点都可以不去除
    但使用的模型本来就是比较简单的n-gram，如果不去除一些标点会导致生成的文本很乱

    :param folder_path: 停用词路径，这里就是同目录下的self_stopwords.txt
    :return: 存放了停用词的集合 stopwords
    """
    stopwords = set()
    with open(folder_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    for line in lines:
        stopwords.add(line.strip())
    return stopwords


def read_reports(folder_path):
    """
    读取文件夹下所有文件

    :param folder_path:存放txt文件的文件夹路径，这里就是同目录下的reports文件夹
    :return:存放文件内容的列表 documents
    """
    documents = []

    for report_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, report_name)
        if os.path.isfile(file_path) and report_name.endswith('.txt'):
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
                documents.append(text)
    return documents


def extract_triples(documents):
    """
    提取文档中的三元组
    对 read_reports 函数返回的 documents 列表进行处理

    :param documents: read_reports 函数返回的 documents 列表
    :return: 存放了三元组的列表 triples
    """
    triples = []
    # 遍历文档,对文档进行分词
    for document in documents:
        # 加载停用词与自定义词典
        jieba.load_userdict('self_userdict.txt')
        stopwords = stopwords_load('self_stopwords.txt')
        stopwords.add('\n')
        stopwords.add(' ')
        stopwords.add('\u2002')

        words = jieba.lcut(document)
        processed_words = []

        for word in words:
            if word not in stopwords:
                processed_words.append(word)

        # 提取三元组
        for i in range(len(processed_words) - 2):
            triples.append((processed_words[i], processed_words[i + 1], processed_words[i + 2]))
    return triples


def calculate_last_word_probability(triplets_list):
    """
    计算三元组中，末尾词在前两个词出现时的概率

    :param triplets_list: 传入的是 extract_triples 函数的返回值 triples
    :return: 存放了末尾词在前两词出现时的概率的字典 probabilities
    """
    # 创建字典来存储每个二元组出现的频率
    bigrams_count = {}
    for triplet in triplets_list:
        # 获取前两个词组成的二元组
        bigram = (triplet[0], triplet[1])
        # 如果二元组不在字典中，添加并初始化计数为0
        if bigram not in bigrams_count:
            bigrams_count[bigram] = {"total_count": 0}
        # 更新二元组的出现次数
        bigrams_count[bigram]["total_count"] += 1
        # 如果当前三元组的最后一个词在前两个词出现时，更新最后一个词的计数
        if triplet[2] in bigrams_count[bigram]:
            bigrams_count[bigram][triplet[2]] += 1
        else:
            bigrams_count[bigram][triplet[2]] = 1

    # 计算每个三元组中最后一个词在前两个词出现时的概率
    # 范围(0-1]
    probabilities = {}
    for bigram, counts in bigrams_count.items():
        total_count = counts["total_count"]
        for last_word, count in counts.items():
            if last_word != "total_count":
                probability = (count / total_count)
                probabilities[(bigram[0], bigram[1], last_word)] = probability
    print("概率计算完毕！")
    return probabilities


def generate_text_with_probability(starting_ngram, probabilities, num_words):
    """
    初次循环时使用初始二元组找出在此二元组后有概率出现的词
    之后使用末尾两个词作为二元组找出二元组后有概率出现的词，按照权重随机选择
    如果找不到，那么随机选择一个词加入到文本中

    :param starting_ngram: 需要给定初始的二元组
    :param probabilities:
    :param num_words:
    :return:
    """
    text = list(starting_ngram)
    print("\n打印文本生成过程的选择的词：")
    for _ in range(num_words - len(starting_ngram)):
        last_ngram = tuple(text[-len(starting_ngram):])
        next_word_probabilities = {ngram: prob for ngram, prob in probabilities.items() if ngram[:len(last_ngram)] == last_ngram}

        print(next_word_probabilities)

        if next_word_probabilities:
            next_word = random.choices(list(next_word_probabilities.keys()), weights=list(next_word_probabilities.values()))[0][-1]

            print(next_word)

            text.append(next_word)
        else:
            # 如果找不到匹配的三元组，则随机选择一个词
            next_word = random.choice(list(probabilities.keys()))[-1]

            print(next_word)

            text.append(next_word)
    return ''.join(text)


def split_text(generated_text, n):
    """
    将生成的文本按字数 n 划分为段落
    否则直接写入txt文件会是一长串，很难看

    :param generated_text: 生成的文本
    :param n: 指定一个段落的字数
    :return: 分段后的段落列表
    """
    return [generated_text[i:i + n] for i in range(0, len(generated_text), n)]


def write_to_file(segments, filename):
    """
    将划分后的段落写入文件

    :param segments: 分段后的段落列表
    :param filename: 要写入的文本文件名
    :return: 无返回
    """
    with open(filename, 'w', encoding='utf-8') as file:
        for segment in segments:
            file.write(segment + '\n')