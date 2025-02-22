from server.LabelFunction.LF_by_form import spanize_and_label_text_by_text,basic_text_sentiment_classification_for_sentence
from server.textprocess.textProcessor import get_all_token_sets

from server.LabelFunction.LF_helper import match_label_for_form_with_rules,get_label_for_form,sentiment_labels,rules
from server.utils.helper import label_string2int
import re
import server.utils.config as config
#辅助函数，将原文转化为标签列表形式
def text_to_form_list(text,alltokensets,aspect):
    spanized_text_dict=spanize_and_label_text_by_text(text,alltokensets)
    #提取出标签组成的句型列表，没有标签的span用none代替
    form_list=[]
    for i, item in enumerate(spanized_text_dict["span_list"]):
        if item["span"]==aspect:
            form_list.append("aspect")
        elif item["label"]!="":
            form_list.append(item["label"])
        else:
            continue
            #form_list.append("none")
    return form_list

#辅助函数  如果direction=="near"返回原文本，如果direction=="forward"则截取文本在aspect及其之前的文本，如果direction=="backward"则截取文本在aspect及其之后的文本，
def extract_directional_text(text, aspect, direction):
    # 使用正则表达式将文本分割为单词
    words = re.findall(r'\S+', text)  # \S+ 匹配任何非空白字符的单词
    
    if aspect not in words:
        return text  # 如果 aspect 不在文本中，返回空字符串
    
    # 找到 aspect 在文本中的索引
    aspect_index = words.index(aspect)
    
    if direction == "forward":
        # 截取从 aspect 及其之前的部分
        context = words[:aspect_index + 1]  # 包括 aspect 本身
    
    elif direction == "backward":
        # 截取从 aspect 及其之后的部分
        context = words[aspect_index:]  # 包括 aspect 本身

    else:
        return text  # 如果 direction 不是 "forward" 或 "backward"，返回空字符串

    # 将截取的单词列表拼接成新的文本，并返回
    return ' '.join(context)

#找到离aspect最近的情感词作为该aspect的情感，考虑到negation和contrast
def label_with_token_match_method(text,aspect,direction,alltokensets,sentiment_labels,rules): 
    handled_text=extract_directional_text(text, aspect, direction)
    spanized_text_dict=spanize_and_label_text_by_text(handled_text,alltokensets)

    # 找到 aspect 的下标
    aspect_index = next((i for i, item in enumerate(spanized_text_dict["span_list"]) if item['span'] == aspect), None)
    #提取出标签组成的句型列表，没有标签的span用none代替 
    form_list=text_to_form_list(handled_text,alltokensets,aspect)

    #print("origin list:",form_list)
    if aspect_index is None:
        return config.NEUTRAL  # 如果未找到 aspect
    
    #将匹配规则应用在当前句子中
    matched_form=match_label_for_form_with_rules(form_list,rules)
    #print ("matched list:",matched_form)

    #找到aspect的坐标
    aspect_idx= matched_form.index("aspect")
    # 初始化最近情感标签的距离和值
    closest_sentiment = "none"
    closest_distance = float("inf")

    # 遍历列表，查找情感标签
    for i, label in enumerate(matched_form):
        if label in sentiment_labels:
            # 计算当前情感标签与目标标签的距离
            distance = abs(i - aspect_idx)
            if distance < closest_distance:
                closest_distance = distance
                closest_sentiment = label

    if closest_sentiment=="none":
        return config.NEUTRAL
    else:
        return label_string2int(closest_sentiment)

#辅助函数，从文本中aspect所在位置，截取其前后 window_size个单词，并返回新截取的文本
def extract_context(text, aspect, window_size):
    # 使用正则表达式将文本分割为单词
    words = re.findall(r'\S+', text)  # \S+ 匹配任何非空白字符的单词
    if aspect not in words:
        return ''  # 如果 aspect 不在文本中，返回空字符串

    # 找到 aspect 在文本中的索引
    aspect_index = words.index(aspect)

    # 确定前后窗口范围，确保不会超出文本长度
    start_index = max(0, aspect_index - window_size)
    end_index = min(len(words), aspect_index + window_size + 1)

    # 截取范围内的单词并组成新文本
    context = words[start_index:end_index]
    
    # 返回以空格分隔的新文本
    return ' '.join(context)

#首先定位到aspect,在aspect前后window_range大小的范围进行文本情感分析，用这个范围内的情感作为该aspect的情感
def label_with_window_analysis_method(text,aspect,alltokensets,window_size,sentiment_labels,rules):
    window_text=extract_context(text, aspect, window_size)
    #提取出标签组成的句型列表，没有标签的span用none代替
    form_list=text_to_form_list(window_text,alltokensets,aspect)
    
    #print("origin form",form_list)
    #将匹配规则应用在当前句子中
    matched_form=match_label_for_form_with_rules(form_list,rules)
    #print("aftering matching form",matched_form)
    final_label= get_label_for_form(matched_form, rules, sentiment_labels)

    return label_string2int(final_label)

#使用全文的情感来推测文中aspect的情感，适用于只含有一种aspect的短文（推特评论）
def label_with_sc_method(text,aspect,alltokensets,sentiment_labels,rules):
    #提取出标签组成的句型列表，没有标签的span用none代替
    form_list=text_to_form_list(text,alltokensets,aspect)

    #将匹配规则应用在当前句子中
    matched_form=match_label_for_form_with_rules(form_list,rules)
    final_label= get_label_for_form(matched_form, rules, sentiment_labels)
    return label_string2int(final_label)

if __name__=='__main__':
    aspect='Trump'
    alltokensets=get_all_token_sets()
    text="“The reasons for my departure are personal, but it has been my great honor to serve President Trump and this administration,” Dubke, 47, wrote in an email to friends, according to Politico."
    #print(text_to_form_list(text,alltokensets,aspect))
    window_size=5
    #text2="The atmosphere is unheralded, the service impecible, and the food magnificant."
    print(label_with_token_match_method(text,aspect,"near",alltokensets,sentiment_labels,rules))

    print(label_with_window_analysis_method(text,aspect,alltokensets,window_size,sentiment_labels,rules))
''' 
spanize_and_label_text_by_text的返回结果：
{'span_list': [{'span': 'On the contrary', 'start': 0, 'end': 3, 'label': 'contrast'}...],
'form': ['contrast', 'negation', 'positive', 'positive'], 
'label': 'neutral'
'''