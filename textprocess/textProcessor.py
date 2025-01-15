import json
import os
import pandas as pd
from server.utils.helper import Json2Dict
from server.utils.helper import Dict2Json
import spacy
from server.utils.helper import get_correct_path
import server.utils.config as config
# 在代码的开头加载模型
nlp = spacy.load("en_core_web_lg")
def dataloader(file_path="../"+config.train_dataset_path):#从本地数据集读取数据，list形式提取所有text
    current_dir = os.path.dirname(os.path.abspath(__file__))  # 当前脚本所在目录
    full_path = os.path.join(current_dir, file_path)  # 拼接成完整路径
    df = pd.read_csv(full_path)
    return df, df['text'].to_list()

def get_all_token_sets(file_path="../datasets/tokensets/all-token-sets.json"):#得到当前所有的token_sets
    current_dir = os.path.dirname(os.path.abspath(__file__))  # 当前脚本所在目录
    full_path = os.path.join(current_dir, file_path)  # 拼接成完整路径
    all_token_sets = Json2Dict(full_path)  # 将文件内容加载为字典{"positive":[word1,word2..wordn], "negative":[word1,word2..wordn]}
    return all_token_sets

def get_custom_token_sets(label_list=[]):#提取前端传来的用户选择的某个tokenset=["label1","label2"]中提到标签对应的word list
    all_token_sets=get_all_token_sets()
    new_token_sets = {label: all_token_sets.get(label, []) for label in label_list}
    return new_token_sets

def update_all_token_sets(new_added_tokens={}, file_path="../datasets/tokensets/all-token-sets.json"):
    # new_added_tokens形如 {"label1": ["token1", "token2"], "label2": ["token3"]}
    current_dir = os.path.dirname(os.path.abspath(__file__))  # 当前脚本所在目录
    full_path = os.path.join(current_dir, file_path)  # 拼接完整路径

    # 加载现有 tokensets
    if not os.path.exists(full_path):
        print(f"File not found: {full_path}. Initializing an empty token set.")
        all_token_sets = {}
    else:
        all_token_sets = Json2Dict(full_path)

    # 遍历新添加的 tokens
    for new_label, new_tokens in new_added_tokens.items():
        for token in new_tokens:
            # 从其他标签中移除当前 token
            for existing_label, existing_tokens in all_token_sets.items():
                if existing_label != new_label and token in existing_tokens:
                    existing_tokens.remove(token)
                    print(f"Removed token '{token}' from label '{existing_label}'.")

        # 添加 token 到新标签
        if new_label in all_token_sets:
            initial_count = len(all_token_sets[new_label])
            all_token_sets[new_label] = list(set(all_token_sets[new_label]).union(new_tokens))
            print(f"Updated label '{new_label}': added {len(all_token_sets[new_label]) - initial_count} new tokens.")
        else:
            all_token_sets[new_label] = new_tokens
            print(f"Added new label '{new_label}' with {len(new_tokens)} tokens.")

    # 保存修改后的 tokensets
    Dict2Json(full_path, all_token_sets)


def replace_all_token_sets(new_tokensets={}, file_path="../datasets/tokensets/all-token-sets.json"):#使用前端传来的tokensets直接替换原有的tokensets
    #new_added_tokens形如 {"label1": ["token1", "token2"], "label2": ["token3"]}
    current_dir = os.path.dirname(os.path.abspath(__file__))  # 当前脚本所在目录
    full_path = os.path.join(current_dir, file_path)  # 拼接完整路径
    Dict2Json(full_path, new_tokensets)

def get_label_for_word(word):#从all_token_sets中查找该词的标签
    all_token_sets=get_all_token_sets()
    for key, word_list in all_token_sets.items():
        # 检查 word 是否在当前键的列表中
        if word in word_list:
            return key
    # 如果没有找到，返回 "none"
    return "none"

def tokenize_text_by_sentence(text):#未使用#将文本按句切割，每一句是多个token_dict组成的list
    tokenized_text_list=[]
    doc = nlp(text)#spacy文本处理
    for sent in doc.sents:  # 遍历每个句子
        sentence_tokens = []  # 用于存储当前句子的所有 token_dict
        for token in sent:  # 遍历句子中的每个词（token）
            # 创建 token_dict，包含词文本、词性和标签
            token_dict = {
                "text": token.text,
                "partofspeech":token.pos_,#词性
                "label": get_label_for_word(token.text)  # 使用实体标签作为示例（可替换）
            }
            sentence_tokens.append(token_dict)  # 将 token_dict 添加到当前句子列表中
        
        tokenized_text_list.append(sentence_tokens)  # 将句子列表添加到结果的 "Text" 中
    
    return tokenized_text_list



'''tokenized_text_list返回形式:
text = "This is the first sentence.
tokenized_text_list=
    [
        [   代表第一句话，其中每句话以token_list的形式存储，每个token以字典形式存储了该token的内容，词性，标签
            {"text": "This",  "partofspeech":"","label": ""},
            {"text": "is", "partofspeech":"","label": ""},
            {"text": "the", "partofspeech":"","label": ""},
            {"text": "first", "partofspeech":"","label": ""},
            {"text": "sentence", "partofspeech":"","label": ""},
            {"text": ".", "partofspeech":"","label": ""}
        ],
        [第二句话
        ]
    ]
'''

def get_text_byID(idx):#按照id返回对应text处理后的结果
    df, texts = dataloader(get_correct_path("../"+config.train_dataset_path))
    text = texts[idx]
    return text


if __name__=='__main__':
    text="In a matter of days US President Joe Biden's administration and Russia have made separate - but significant - moves aimed at influencing the outcome of the war in Ukraine."
    tokenized_text_list=tokenize_text_by_sentence(text)
    for  token_dict in tokenized_text_list[0]:
        print(token_dict["text"],"(",token_dict["partofspeech"],")",end=' ')
