import json
import pandas as pd
import os
import re
import requests
from server.LabelFunction.LF_by_svo_extract import extract_svo,label_with_svo_method,filtered_svo_with_aspect
from server.LabelFunction.LF_by_token import label_with_token_match_method,label_with_sc_method
from nltk.tokenize import sent_tokenize
import server.utils.config as config
import server.LabelFunction.selfLabelFunc

posnews_path="../datasets/positive_news"
negnews_path="../datasets/negative_news"
#旧新闻提取：从积极消极各自1000个json文件中提取出与aspect有关的句子————————————————
def csv2list():
    # 读取CSV文件
    file_path = '../datasets/restaurants-train.csv'  # 替换为你的CSV文件路径
    current_dir = os.path.dirname(os.path.abspath(__file__))  # 当前脚本所在目录
    full_path = os.path.join(current_dir, file_path)  # 拼接成完整路径
    
    data = pd.read_csv(full_path)
    
    # 提取'text'列
    if 'text' in data.columns:
        text_column = data['text']
        text_list = text_column.tolist()
        return text_list
    else:
        return False


def extract_text_and_polarity():
    file_path = '../'+config.test_dataset_path  # 替换为你的CSV文件路径
    current_dir = os.path.dirname(os.path.abspath(__file__))  # 当前脚本所在目录
    full_path = os.path.join(current_dir, file_path)  # 拼接成完整路径
    data = pd.read_csv(full_path)
    # 检查是否同时包含 'text' 和 'polarity' 列
    if 'text' in data.columns and 'polarity' in data.columns:
        # 提取 'text' 和 'polarity' 列
        text_column = data['text']
        polarity_column = data['polarity']

        # 将两列组合成字典列表
        result = [{'text': text, 'polarity': polarity} for text, polarity in zip(text_column, polarity_column)]
        return result
    else:
        # 如果列不存在，返回 False
        return False

#从文本中提取与aspect相关的句子
def find_sentences_with_aspect(text, aspect):
    # 使用正则表达式分割文本为句子（以句号、问号或感叹号作为分隔符）
    #sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = sent_tokenize(text)
    aspect = aspect.lower()
    matching_sentences = [
        sentence for sentence in sentences
        if aspect in sentence.lower()
    ]
    
    return matching_sentences

#读取指定文件夹中的所有 JSON 文件，提取与指定 aspect 相关的文本及其文件名。
def get_aspect_texts_from_json(file_path, aspect="Trump", label="Positive"):
    current_dir = os.path.dirname(os.path.abspath(__file__))  # 当前脚本所在目录
    full_path = os.path.join(current_dir, file_path)  # 拼接成完整路径
    extracted_data = []
    if not os.path.exists(full_path):
        print(f"目录 '{full_path}' 不存在。")
        return []

    for index, file in enumerate(os.listdir(full_path), start=1):
        if index % 100 == 0:
            print(f"{index} files processed!")

        if file.endswith('.json'):
            jsonfile_path = os.path.normpath(os.path.join(full_path, file))
            with open(jsonfile_path, 'r', encoding='utf-8') as f:
                news_dict = json.load(f)
                text = news_dict.get('text', '')
                if aspect.lower() in text.lower() and news_dict.get("language") == "english":
                    extracted_data.append({
                        'FileName': file,
                        'Text': text,
                        'Label': label
                    }) 

    print(f"从文件夹 '{file_path}' 中找到 {len(extracted_data)} 条与 '{aspect}' 相关的文本。")
    return extracted_data

def find_sentences_with_string(text, target_string, case_insensitive=True):
    sentences = sent_tokenize(text)  # 按句子分割
    if case_insensitive:
        target_string = target_string.lower()
        matching_sentences = [
            sentence for sentence in sentences if target_string in sentence.lower()
        ]
    else:
        matching_sentences = [
            sentence for sentence in sentences if target_string in sentence
        ]
    return matching_sentences

#读取指定文件夹中的所有 JSON 文件，提取与指定 aspect 相关的句子及其文件名。
def get_aspect_sentences_from_json(file_path, aspect="Trump", label="Positive"):
    current_dir = os.path.dirname(os.path.abspath(__file__))  # 当前脚本所在目录
    full_path = os.path.join(current_dir, file_path)  # 拼接成完整路径
    extracted_data = []

    if not os.path.exists(full_path):
        print(f"目录 '{full_path}' 不存在。")
        return []

    for index, file in enumerate(os.listdir(full_path), start=1):
        if index % 100 == 0:
            print(f"{index} files processed!")

        if file.endswith('.json'):
            jsonfile_path = os.path.normpath(os.path.join(full_path, file))
            with open(jsonfile_path, 'r', encoding='utf-8') as f:
                news_dict = json.load(f)
                text = news_dict.get('text', '')

                # 查找包含 aspect 的句子
                if news_dict.get("language") == "english":
                    matching_sentences = find_sentences_with_string(text, aspect)
                    for sentence in matching_sentences:
                        extracted_data.append({
                            'id': file,
                            'text': sentence,
                            'term':aspect,
                            'polarity': label.lower(),
                            'label':'neutral'
                        })

    print(f"从文件夹 '{file_path}' 中找到 {len(extracted_data)} 条与 '{aspect}' 相关的句子。")
    return extracted_data


#从新闻中提取包含指定 aspect 的文本，保存到 CSV 文件。
def construct_data_sets(aspect):
    file_path = '../datasets/test_news_output.csv'  # 替换为你的 CSV 文件路径
    file_path2 = '../datasets/test_news_sentence.csv'  # 替换为你的 CSV 文件路径
    current_dir = os.path.dirname(os.path.abspath(__file__))  # 当前脚本所在目录
    dataset_path = os.path.join(current_dir, file_path2)  # 拼接成完整路径

    # 提取正面和负面文本
    #pos_texts = get_aspect_texts_from_json(posnews_path, aspect, label="Positive")
    pos_texts = get_aspect_sentences_from_json(posnews_path, aspect, label="Positive")
    neg_texts = get_aspect_sentences_from_json(negnews_path, aspect, label="Negative")
    all_texts = pos_texts[:150] + neg_texts[:150]

    for index, data in enumerate(all_texts):
        data['id'] = index
    # 创建 DataFrame 并保存到 CSV
    df = pd.DataFrame(all_texts)
    df.to_csv(dataset_path, index=False, encoding='utf-8-sig')
    print(f"数据集已保存到: {dataset_path}")



#新·新闻提取：从jsonl文件中提取出与aspect有关的句子————————————————
#从url中下载json文件
def  url_download():
    url = "https://raw.githubusercontent.com/fhamborg/NewsMTSC/6b838e00f54423c253806327a0ae24dbffa24c9e/NewsSentiment/experiments/default/datasets/newsmtsc-rw-hf/test.jsonl"
    file_path = "test.jsonl"  # 保存的文件名

    response = requests.get(url)

    if response.status_code == 200:
        with open(file_path, "wb") as file:
            file.write(response.content)
        print(f"文件下载成功: {file_path}")
    else:
        print(f"下载失败，状态码: {response.status_code}")

#jsonl文件转csv
def jsonl2csv(aspect,train_or_test):
    current_dir = os.path.dirname(os.path.abspath(__file__))  # 当前脚本所在目录
    # 读取 JSONL 文件
    if train_or_test=="train":
        input_file = os.path.join(current_dir, '../datasets/11k_news_sentence/train.jsonl')  # 你的 JSONL 文件路径
        output_file = os.path.join(current_dir, '../datasets/11k_news_sentence/news11k_train.csv')  # 输出的 CSV 文件路径
    else:
        input_file = os.path.join(current_dir, '../datasets/11k_news_sentence/test.jsonl')  # 你的 JSONL 文件路径
        output_file = os.path.join(current_dir, '../datasets/11k_news_sentence/news11k_test.csv')  # 输出的 CSV 文件路径

    # 存储提取的数据
    extracted_data = []

    # 解析 JSONL 文件
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            news_dict = json.loads(line.strip())  # 解析 JSON
            mention = news_dict.get("mention", "")

            # 只保留 mention 为 "Trump" 的数据
            if mention.lower() == aspect.lower():
                extracted_data.append({
                    "id": news_dict.get("id", ""),  # ID
                    "text": news_dict.get("sentence", ""),  # 改名
                    "aspect": mention,  # 改名
                    "polarity": ('positive' if news_dict.get("polarity", "")==1 else ('negative' if news_dict.get("polarity", "")==0 else 'neutral')),  # 情感
                    "label":"neutral"
                })

    # 转换为 DataFrame
    if train_or_test=="train":
        df = pd.DataFrame(extracted_data[:450])
    else:
        df = pd.DataFrame(extracted_data[:150])

    # 保存为 CSV 文件
    df.to_csv(output_file, index=False, encoding="utf-8")

    print(f"提取完成，符合条件的数据已保存到: {output_file}")

if __name__=="__main__":
    #construct_data_sets("Trump")
    jsonl2csv("Trump","test")
