from server.textprocess.textProcessor import get_all_token_sets
from server.textprocess.textSpanizer import gain_span_list

import os
import json
import spacy
from spacy import displacy
nlp = spacy.load("en_core_web_lg")
#存取所有句型--已废弃
def save_labeling_functions(labelfuncs, file_path="./SavedLabelFunc.json"):#保存句型
    current_dir = os.path.dirname(os.path.abspath(__file__))  # 当前脚本所在目录
    full_path = os.path.join(current_dir, file_path)  # 拼接成完整路径
    json_labelfuncs=json.dumps(labelfuncs)
    with open(full_path, "w") as file:
        file.write(json_labelfuncs)

def load_labeling_functions(file_path="./SavedLabelFunc.json"):#加载句型
    current_dir = os.path.dirname(os.path.abspath(__file__))  # 当前脚本所在目录
    full_path = os.path.join(current_dir, file_path)  # 拼接成完整路径
    with open(full_path, "r") as file:
        labeling_functions = json.load(file)
    return labeling_functions

def update_labeling_functions(form_label_dict):#更新句型
    # 加载当前的标记函数
    curr_labeling_functions = load_labeling_functions()
    # 检查是否已经存在相同的 form
    for lf in curr_labeling_functions:
        if lf["form"] == form_label_dict["form"]:
            # 如果 form 存在，更新其 label
            lf["label"] = form_label_dict["label"]
            break
    else:
        # 如果 form 不存在，添加整个 form_label_dict
        curr_labeling_functions.append(form_label_dict)

    # 保存更新后的标记函数
    save_labeling_functions(curr_labeling_functions)



#标记函数，根据标记函数得到每句的标签。
def basic_text_sentiment_classification_for_sentence(form):#在没有LF模板下，默认根据pos,neg,negation,contrast四个标签为该句标记
    #form=['contrast','positive','negative']
    positive_count = 0
    negative_count = 0
    i = 0

    while i < len(form):
        curr_label = form[i]
        
        if curr_label == "positive":
            positive_count += 1
        
        elif curr_label == "negative":
            negative_count += 1
        
        elif curr_label == "negation" and i + 1 < len(form):
            next_label = form[i + 1]
            if next_label == "positive":
                negative_count += 1
            elif next_label == "negative":
                positive_count += 1
            
            i += 1  # Skip the next element as it's already processed
        
        elif curr_label == "contrast":
            positive_count = 0
            negative_count = 0
        
        i += 1  # Move to the next element
    
    if(positive_count>negative_count):
        return "positive"
    elif(positive_count<negative_count):
        return "negative"
    else:
        return "neutral"

def is_subsequence(labeling_function_form, text_form):#标记函数的句型是否是该句句型的递增子序列（即该句是否含有这个句型）
    pos = 0
    for elem in labeling_function_form:
        try:
            pos = text_form.index(elem, pos) + 1
        except ValueError:
            return False
    return True

def sentiment_classification_for_sentence_loose(text_form,all_labeling_functions):#松散型：只要该句的递增子序列与标记函数相等即刻
    longest_matched_form=[]
    longest_matched_form_label=""
    for label_func in all_labeling_functions:
        if(is_subsequence(label_func["form"],text_form) and len(label_func["form"])>len(longest_matched_form)):#寻找与该句句型匹配的最长标记函数句型
            longest_matched_form=label_func["form"]
            longest_matched_form_label=label_func["label"]
    if(longest_matched_form_label==""):
        longest_matched_form_label=basic_text_sentiment_classification_for_sentence(text_form)
    return longest_matched_form_label

def sentiment_classification_for_sentence_strict(text_form,all_labeling_functions):#严苛型；该句的句型必须和标记函数一模一样
    for label_func in all_labeling_functions:
        if(text_form==label_func["form"]):
            return label_func["label"]
    return basic_text_sentiment_classification_for_sentence(text_form)

def sentiment_classification_for_text(whole_text_list):#根据每句的标签得到整个文本的标签
    pos_count=0 
    neg_count=0
    neutral_count=0
    for sentence_dict in whole_text_list:
        if(sentence_dict["label"]=="positive"):
            pos_count+=1
        elif(sentence_dict["label"]=="negative"):
            neg_count+=1
        else:
            neutral_count+=1
    if(pos_count>neg_count and pos_count>neutral_count):
        return "positive"
    elif(neg_count>pos_count and neg_count>neutral_count):
        return "negative"
    else:
        return "neutral"

def get_selected_text_form(whole_text_list):#将用户扫选的文本（可能不是以句号结尾）中所有的标签组成一个form返回
    form=[]
    for sentence_dict in whole_text_list:
        form.extend(sentence_dict["form"])
    return form

#文本分析
def get_sentence_dict(span_list, all_token_sets):# 标记span_list并得到该句的句型form，以sentence_dict形式返回
    #all_labeling_functions=load_labeling_functions()
    sentence_dict={}
    key_labels = []  # 逐个查找该句中span，如有对应的label，按序存入key_labels中
    for span_dict in span_list:
        span = span_dict["span"]
        # 遍历all_token_sets的每个label和对应的单词列表
        for label, words in all_token_sets.items():
            # 如果span在words列表中，找到对应的key
            if span.lower() in [word.lower() for word in words]:
                span_dict["label"]=label
                key_labels.append(label)

    sentence_dict["span_list"]=span_list#该句按单词短语拆分并被标记后得到的span_list
    sentence_dict["form"]=key_labels#该句的句型（形如["contrast","unsure","positive"]）
    #sentence_dict["label"]=sentiment_classification_for_sentence_strict(key_labels,all_labeling_functions)#该句根据标记函数得到的情感标签
    #sentence_dict["label"]=basic_text_sentiment_classification_for_sentence(key_labels)#该句根据标记函数得到的情感标签
    return sentence_dict

def spanize_and_label_text_by_sentence(text):#把文本按句切割；并对每句进行spanize，对句子中的每个span贴上标签，并根据标签返回该句的句型；最后使用列表返回所有句的span_list和句型
    all_token_sets=get_all_token_sets()
    whole_text_list=[]
    doc = nlp(text)#spacy文本处理
    for sent in doc.sents:
        span_list_sentence=gain_span_list(sent.text)
        sentence_dict=get_sentence_dict(span_list_sentence,all_token_sets)
        whole_text_list.append(sentence_dict)
    return whole_text_list

def spanize_and_label_text_by_text(text,all_token_sets):#把整个文本直接切割；并进行spanize，对每个span贴上标签，并根据标签返回整体句型；最后使用字典返回span_list和句型
    span_list=gain_span_list(text)
    text_dict=get_sentence_dict(span_list,all_token_sets)
    return text_dict





# 测试句子
text = "It was Manchester City, the great domestic power of the modern era, who found themselves trapped in this perfect storm, surely tossed and blown out of contention for a fifth successive Premier League title."
text2="It was Manchester City, the great domestic power of the modern era, who found Manchester City, the great domestic power of the modern era trapped in this perfect storm, surely tossed and blown out of contention for a fifth successive Premier League title."
text3="His motto was everyone was his brother or sister and sharing support, wisdom and the love of the Lord was his responsibility."
text4="The food is top notch, the service is attentive, and the atmosphere is great."
text6="When John Stonehouse hatched his plan to completely disappear, he was a troubled man. "
text6cr="Virtually all of them approved of his cabinet picks, hailing his cabinet picks as much-needed disruptors to what they see as a corrupt establishment."

if __name__=='__main__':
    doc=nlp(text6cr)
    displacy.serve(doc, style="dep")
'''
    whole_text_list=analyze_text(text)
    for sentence in whole_text_list:
        for item in sentence["span_list"]:
            print(item['span'],"(",item['label'],")",end=" ")
        print(" ")

        for label in sentence["form"]:
            print(label,end=" ")
        print(" ")

        print(sentence["label"])

    label=sentiment_classification_for_text(whole_text_list)
    print("The final label of the text is",label)

    text_dict=spanize_and_label_text_by_text(text)
    for ele in text_dict["span_list"]:
        print(ele)
''' 
