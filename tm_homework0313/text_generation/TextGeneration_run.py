"""
请根据第1次作业爬取的政府工作报告的内容，列举所有的二元组（bigram），并根据二元组生成一个1000字左右的文本。

要求：

（1）源代码

（2）代码运行中间过程截图

（3）结果文件

（4）请将本次作业的所有内容打成以zip格式打包，并以"姓名"命名

进阶需求：鼓励用三元组完成
"""
import TextGeneration_tools

documents = TextGeneration_tools.read_reports("reports")

# 提取三元组
all_triples = TextGeneration_tools.extract_triples(documents)

print(f"提取三元组的数量:{len(all_triples)}")
probabilities = TextGeneration_tools.calculate_last_word_probability(all_triples)

# 给定的开头
starting_ngram = ("国务院","总理")

# 生成文本，因为生成文本是按词数计算的，所以600词大概在1000字
generated_text = TextGeneration_tools.generate_text_with_probability(starting_ngram, probabilities, 600)

print(f"生成的文本字数为:{len(generated_text)}")
print("\n")
print(generated_text)

# 划分文本
segments = TextGeneration_tools.split_text(generated_text, 100)

# 写入到txt文件中
TextGeneration_tools.write_to_file(segments,'generated_report.txt')
