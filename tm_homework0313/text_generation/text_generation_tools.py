import os
import random
import jieba


# 加载停用词
# 这里的停用词主要是标点符号之类的
def stopwords_load(folder_path):
    stopwords = set()
    with open(folder_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    for line in lines:
        stopwords.add(line.strip())
    return stopwords


def read_reports(folder_path):
    documents = []

    for report_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, report_name)
        if os.path.isfile(file_path) and report_name.endswith('.txt'):
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
                documents.append(text)
    return documents


# 定义一个函数来提取文档中的三元组
def extract_triples(documents):
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
    # 创建字典来存储每个二元组出现的频率
    bigrams_count = {}
    for triplet in triplets_list:
        # 获取前两个词组成的二元组
        bigram = (triplet[0], triplet[1])
        # 如果二元组不在字典中，则添加并初始化计数为0
        if bigram not in bigrams_count:
            bigrams_count[bigram] = {"total_count": 0}
        # 更新二元组的总出现次数
        bigrams_count[bigram]["total_count"] += 1
        # 如果当前三元组的最后一个词在前两个词出现时，更新最后一个词的计数
        if triplet[2] != "last_word_count":
            if triplet[2] in bigrams_count[bigram]:
                bigrams_count[bigram][triplet[2]] += 1
            else:
                bigrams_count[bigram][triplet[2]] = 1

    # 计算每个三元组中最后一个词在前两个词出现时的概率
    probabilities = {}
    for bigram, counts in bigrams_count.items():
        total_count = counts["total_count"]
        for last_word, count in counts.items():
            if last_word != "total_count":
                probability = (count / total_count)
                probabilities[(bigram[0], bigram[1], last_word)] = probability

    return probabilities


def generate_text_with_probability(starting_ngram, probabilities, num_words):
    text = list(starting_ngram)
    for _ in range(num_words - len(starting_ngram)):
        last_ngram = tuple(text[-len(starting_ngram):])
        next_word_probabilities = {ngram: prob for ngram, prob in probabilities.items() if ngram[:len(last_ngram)] == last_ngram}
        if next_word_probabilities:
            next_word = random.choices(list(next_word_probabilities.keys()), weights=list(next_word_probabilities.values()))[0][-1]
            text.append(next_word)
        else:
            break
    return ''.join(text)