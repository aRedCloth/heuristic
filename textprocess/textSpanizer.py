import json
import spacy
import os
from spacy.matcher import Matcher
#初始的spacy会将每个单词独立拆分，不能识别多个单词组成的一个具有独立意义的短语
#所以这个模块使用模式匹配来帮助用来把短语存储成模式

nlp = spacy.load('en_core_web_lg')
matcher = Matcher(nlp.vocab)


def identify_word_tag(tags=["ADJ"]):# 将连续且词性相同的词识别为一个span
    for tag in tags:
        pattern = [
            {"POS": tag, "OP": "+"}
        ]
        matcher.add(tag, [pattern])

def phrase_add_to_patterns(tokensets, file_path="./patterns.json"):
    """
    将前端传入的tokensets中的短语存储为模式匹配。
    tokensets: dict，标签和短语的字典，形如 {"label1": ["phrase1", "phrase2"], "label2": ["phrase3"]}
    """
    # 加载现有 patterns
    current_dir = os.path.dirname(os.path.abspath(__file__))  # 当前脚本所在目录
    full_path = os.path.join(current_dir, file_path)  # 拼接完整路径
    patterns = load_patterns(full_path)

    # 遍历每个标签和对应的短语列表
    for label, phrases in tokensets.items():
        for phrase in phrases:
            # 仅处理包含多个单词的短语
            words = phrase.split()
            if len(words) > 1:
                # 为每个单词生成一个包含 "LOWER" 键的字典
                pattern = [{"LOWER": word.lower()} for word in words]

                # 先遍历所有标签，检查是否已有该 pattern 并从原标签中删除
                for existing_label, existing_patterns in patterns.items():
                    # 如果标签不是当前标签，且模式已经在该标签的列表中
                    if existing_label != label and pattern in existing_patterns:
                        existing_patterns.remove(pattern)  # 从原标签中删除该模式
                        break  # 删除后跳出循环

                # 添加到当前标签中，避免重复添加
                if label in patterns:
                    if pattern not in patterns[label]:
                        patterns[label].append(pattern)
                else:
                    patterns[label] = [pattern]

    # 保存更新后的 patterns
    save_patterns(patterns)

def save_patterns(patterns, file_path="./patterns.json"):#保存模式匹配
    current_dir = os.path.dirname(os.path.abspath(__file__))  # 当前脚本所在目录
    full_path = os.path.join(current_dir, file_path)  # 拼接成完整路径
    json_patterns=json.dumps(patterns)
    with open(full_path, "w") as file:
        file.write(json_patterns)

def load_patterns(file_path="./patterns.json"):#加载模式匹配
    current_dir = os.path.dirname(os.path.abspath(__file__))  # 当前脚本所在目录
    full_path = os.path.join(current_dir, file_path)  # 拼接成完整路径
    # 读取并加载 JSON 文件
    with open(full_path, "r") as file:
        patterns = json.load(file)
    return patterns

def load_matcher(patterns, matcher):# 将patterns中每项模式添加到matcher中
    for key, pattern_list in patterns.items():
        matcher.add(key, pattern_list)

def identify_matches(doc):#对当前nlp处理后的文本使用matcher进行模式匹配，得到（去重后的）匹配结果
    # 在doc上调用matcher
    matches = matcher(doc)
    span_list = []
    # 遍历所有的匹配结果
    for match_id, start, end in matches:
        string_id = nlp.vocab.strings[match_id]
        span = doc[start:end].text
        span_list.append({"span": span, "start": start, "end": end, "label": string_id})

    return unique_span_list(span_list)#返回去重后的匹配结果

def unique_span_list(span_list):# 匹配结果唯一化，去掉相邻重复项
    i = 0
    while i < len(span_list) - 1:
        # 检测到重复项
        if span_list[i]["start"] == span_list[i + 1]["start"] or span_list[i]["end"] == span_list[i + 1]["end"]:
            if len(span_list[i]["span"]) > len(span_list[i+1]["span"]):
                span_list[i+1] = span_list[i]
            del span_list[i]
        else:
            i += 1
    return span_list

def gain_span_list(text):  #获得 unlabeled_span_list
    doc = nlp(text)
    patterns = load_patterns()
    load_matcher(patterns, matcher)#把保存的模式加载到matcher
    span_list = identify_matches(doc)  # 使用存储的模式patterns对当前被按单词拆分的span_list中的短语整合，获得匹配成功的短语 span_list

    seen_tokens = set()  # 用于存储已经添加的 token 的位置（token.i）

    for match in span_list:  # 先将匹配的短语添加到 seen_tokens 和 span_list 中
        # match["span"] 是整个短语
        start = match["start"]
        end = match["end"]
        # 将此短语中的每个 token 的位置加入 seen_tokens
        for token in doc[start:end]:
            seen_tokens.add(token.i)#存储位置索引，确保能够将一个短语唯一存储

    # 遍历每个 token，添加到 span_list
    for i, token in enumerate(doc):
        # 如果 token 已经存在于 seen_tokens 中，则跳过
        if i in seen_tokens:
            continue
        # 创建字典存储当前 token 的值和位置信息
        #token_dict = {"span": token.text, "start": i, "end": i + 1, "label": ""}
        token_dict = {"span": token.text, "start": i, "end": i + 1, "label": ""}

        # 将字典添加到列表中
        span_list.append(token_dict)
        # 将 token 的位置记录到 seen_tokens 中
        seen_tokens.add(i)

    # 将所有 span 按原文顺序重新排列
    sorted_list = sorted(span_list, key=custom_sort)
    span_list = unique_span_list(sorted_list)
    return span_list

def custom_sort(item):# 排序函数
    return item['start'], len(item['label'])


if __name__=='__main__':
    '''
    patterns = {
        "contrast":[
                    [{"LOWER": "on"}, {"LOWER": "the"}, {"LOWER": "contrary"}],
                    [{"LOWER": "as"}, {"LOWER": "well"}, {"LOWER": "as"}],
                    [{"LOWER": "in"}, {"LOWER": "addition"}]
        ],
        "unsure":[
                    [{"LOWER": "not"}, {"LOWER": "sure"}]
        ]
    }
    span_list
    {'span': 'On the contrary', 'start': 0, 'end': 3, 'label': 'contrast'}
    {'span': ',', 'start': 3, 'end': 4, 'label': ''}
    {'span': 'i', 'start': 4, 'end': 5, 'label': ''}
    {'span': 'am', 'start': 5, 'end': 6, 'label': ''}
    {'span': 'not sure', 'start': 6, 'end': 8, 'label': 'unsure'}
    {'span': 'if', 'start': 8, 'end': 9, 'label': ''}
    {'span': 'John', 'start': 9, 'end': 10, 'label': ''}
    {'span': 'is', 'start': 10, 'end': 11, 'label': ''}
    {'span': 'a', 'start': 11, 'end': 12, 'label': ''}
    {'span': 'good', 'start': 12, 'end': 13, 'label': ''}
    {'span': 'man', 'start': 13, 'end': 14, 'label': ''}
    )
    patterns=load_patterns()

    test="On the contrary, i am not sure if John is a good man. What is your idea, Jonny?"
    span_list=gain_span_list(test)
    for token_dict in span_list:
        print(token_dict)
    '''