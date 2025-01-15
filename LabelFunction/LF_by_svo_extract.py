import spacy
import os
from server.textprocess.textProcessor import get_all_token_sets
from server.textprocess.textSpanizer import gain_span_list
nlp = spacy.load("en_core_web_lg")
from server.utils.helper import label_string2int
import server.utils.config as config
from server.LabelFunction.LF_helper import match_label_for_form_with_rules,get_label_for_form,rules,sentiment_labels
from server.LabelFunction.LF_by_form import spanize_and_label_text_by_text
from collections import Counter

def sentiment_reverse(label):
    if(label=="positive"):
        return "negative"
    elif(label=="negative"):
        return "positive"
    else:
        return label

# 从句子中提取所有 Subject-Verb-Object 结构  Subject-aux(be动词)-
def extract_svo(origin_text):
    text=origin_text.lower()
    doc = nlp(text)
    svos = []
    verb_exist_flag=0 #检测该句中是否有动词
    #inline-function-----------------------------------------------------------
    def expand_phrase(token):#得到主语和宾语中的完整的名词短语，例如:由water得到a bottle of water
        """Expand a token into a full phrase by including modifiers and conjunctions."""
        words = [token.text]
        
        # Collect all children of the token, including modifiers, compounds, and conjunctions
        for child in token.children:
            if child.dep_ in {"amod", "compound", "det", "nummod",  "conj", "attr","poss"}:
                words.append(child.text)
            elif child.dep_ in {"prep"}:# prep+pobj例如 of water
                words.append(child.text)
                for c in child.children:
                    if(c.dep_=="pobj"):
                        words.append(c.text)
        
        # Ensure proper order of words in the phrase by sorting by the token's position
        words_sorted = sorted(words, key=lambda word: doc.text.find(word))
        return " ".join(words_sorted)

    def get_full_verb(main_verb):#得到具有完整意义的动词和动词短语，例如:由blow得到blow out of
        full_verb_unsorted=[main_verb.text]
        # Include prepositions to form verb phrases (e.g., "blown out of")
        for child in main_verb.children:
            if child.dep_ == "prep" or child.dep_ == "aux": #do n't like中的do属于aux, #blow out of中的out和of都属于prep  #advmod是属于这个动词的副词extremely hate
                full_verb_unsorted.append(child.text)

        # Handle negation by appending to the verb (e.g., "don't like")
        neg = next((child.text for child in main_verb.children if child.dep_ == "neg"), None)
        if neg:
            full_verb_unsorted.append(neg)
        full_verb_sorted = sorted(full_verb_unsorted, key=lambda word: doc.text.find(word))
        return " ".join(full_verb_sorted)

    def add_svo(subjects=[],full_verb="",objects=[],svos=[]):#将当前动词以及和他相关的所有主语宾语组成一个完整的svo，放入svos中
        # Add SVO triples
        if subjects and objects:
            for subj in subjects:
                for obj in objects:
                    svos.append([subj, full_verb, obj])
        elif subjects:
            for subj in subjects:
                svos.append([subj, full_verb, " "])
        elif objects:
            for obj in objects:
                svos.append([" ", full_verb, obj])
        # Check if there are any relative clauses (e.g., "who found themselves")
        for child in main_verb.children:
            if child.dep_ == "relcl":
                subjects.append(expand_phrase(child))
                svos.append([full_verb, full_verb, child.text])


    #main-code-----------------------------------------------------------
    for token in doc:
        if token.pos_ in {"ROOT", "VERB", "AUX"}:
            main_verb = token
            full_verb=get_full_verb(main_verb)
            verb_exist_flag=1
            # Initialize subjects and objects lists
            subjects = []
            objects = []

            # 添加verb+object类型
            for child in main_verb.children:
                # Find subjects
                if child.dep_ in {"nsubj", "nsubjpass"}:
                    subjects.append(expand_phrase(child))
                # Find direct objects or predicate nominatives
                if child.dep_ in {"dobj", "attr", "pobj","acomp"}:
                    objects.append(expand_phrase(child))

                # Handle prepositional phrases      e.g.: he sit "on the desk"
                if child.dep_ == "prep":
                    for pobj in child.children:
                        if pobj.dep_ == "pobj":
                            objects.append(expand_phrase(pobj))


            # Add SVO triples
            add_svo(subjects,full_verb,objects,svos)

    return svos

#筛选出含有aspect的svo
def filtered_svo_with_aspect(svos,aspect):
    filtered_svos = []
    for svo in svos:
        for string in svo:
            if aspect.lower() in string.lower():
                filtered_svos.append(svo)
                continue
    return filtered_svos

#为svo中的一个part贴标签，例：["i","was n't","a girl"]中的一个字符串"was n't"
def get_label_for_phrase(phrase, all_token_sets,aspect,rules):
    label_list = []

    # 分割短语为单词并匹配标签
    phrase_words = phrase.split()
    for word in phrase_words:
        for label, words in all_token_sets.items():
            if word.lower() in map(str.lower, words):
                label_list.append(label)
    
    #得到该part的form，形如["negation","positive"]
    #print(phrase,label_list)

    #根据rules得到该form的标签， 得到的结果是一个列表["label1","label2"]
    #但因为该part只是svo中的一部分，所以通常情况下只会得到一个标签
    form_label=match_label_for_form_with_rules(label_list,rules)
    #print("每个part及其句型：",phrase,form_label)

    if form_label:
        return form_label[0]
    else:
        return "neutral"  

#对筛选后的句子进行情感分析
def get_label_for_aspect_in_svos(svos,orgin_aspect,all_token_sets,sentiment_labels,rules):
    aspect=orgin_aspect.lower()
    # 初始化计数器
    label_count = Counter()

    for svo in svos:#对每一个svo进行情感分析
        aspect_pos_index = next((index for index, string in enumerate(svo) if aspect in string), -1) #找到aspect所在的svo的小标 index=0:主语 =1:谓语 =2:宾语
        #print("aspect index:",aspect_pos_index)
        subject_label= get_label_for_phrase(svo[0], all_token_sets,aspect,rules)
        verb_label= get_label_for_phrase(svo[1], all_token_sets,aspect,rules)
        object_label= get_label_for_phrase(svo[2], all_token_sets,aspect,rules)

        #print(svo,":",subject_label,verb_label,object_label)
        if aspect_pos_index!= -1: 
            if verb_label in sentiment_labels:#如果动词能够表达一种情感
                #print("check verb:",verb_label)
                label_count[verb_label] += 3

            else:
                if subject_label in sentiment_labels: #aspect做主语且有对它的情感词
                        label_count[subject_label] += 1

                elif object_label in sentiment_labels:#aspect做宾语且有对它的情感词
                    label_count[object_label] += 1
                else:
                    label_count["neutral"] += 1

        else:#对svo_form进行句型分析，得到情感标签
            svo_form=[subject_label,verb_label,object_label]
            final_label=get_label_for_form(svo_form, rules,sentiment_labels)
            label_count[final_label] += 1
    
    
    del label_count["neutral"]
    # 获取最多的标签及其出现次数
    most_common = label_count.most_common()
    
    #print("label_count:",label_count)

    # 如果没有统计到 sentiment_label，返回 "neutral"
    if not most_common:
        return "neutral"
    max_frequency = most_common[0][1]  # 最大频率
    max_labels = [label for label, freq in most_common if freq == max_frequency]

    # 判断结果
    if len(max_labels) == 1:#有唯一最多的sentiment_label
        return max_labels[0]  # 返回该标签
    else:#有多个最多的标签
        return "neutral"  # 返回 "neutral" 表示

#从text开始的全过程情感分析
def label_with_svo_method(text,aspect,all_token_sets, sentiment_labels,rules):
    final_label=''
    svos=extract_svo(text)
    #print("orgin:",svos)
    filtered_svos=filtered_svo_with_aspect(svos,aspect)#含有aspect的句子
    #print("filtered",filtered_svos)
    if(filtered_svos==[]):#没有动词，直接返回整句的情感
        label_string=get_label_for_phrase(text, all_token_sets,aspect,rules)
        #return label_string2int(label_string)
        final_label = label_string
        return config.ABSTAIN
    
    final_label=get_label_for_aspect_in_svos(filtered_svos,aspect,all_token_sets,sentiment_labels,rules)
    return label_string2int(final_label)


#根据rules对text进行预处理，先化为span_list 再根据规则合并简化
#例如John is not[negation] handsome[positive]->John is negative[negative]
def preprocess_text(text,alltokensets):
    span_list=spanize_and_label_text_by_text(text,alltokensets)["span_list"]
    text_by_word=[]
    text_by_label=[]
    for span in span_list:
        text_by_word.append(span["span"])
        if span["label"] == "":
            text_by_label.append("none")
        else:
            text_by_label.append(span["label"])
    print(text_by_word)
    print(text_by_label)

if __name__=='__main__':
    all_token_sets=get_all_token_sets()
    text="From the appetizers we ate , the dim sum and other variety of foods , it was impossible to criticize the food ."
    text1="the service was pretty poor all around, the food was well below average relative to the cost, and outside there is a crazy bum who harasses every customer who leaves the place."
    text="Food was average and creme brulee was awful - the sugar was charred, not caramelized and smelled of kerosene."
    preprocess_text(text,all_token_sets)
    print(label_with_svo_method(text,"food",all_token_sets, sentiment_labels,rules))
