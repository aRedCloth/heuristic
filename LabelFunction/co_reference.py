import spacy
import os
from fastcoref import spacy_component

nlp = spacy.load("en_core_web_lg")

#配置fastcoref模型，用于共指替换
current_dir = os.path.dirname(os.path.abspath(__file__))  # 当前脚本所在目录
model_path = os.path.join(current_dir, "../models/coreference_model")  # 拼接成完整路径
nlp.add_pipe(
    "fastcoref",
    config={
        "model_architecture": "FCoref",  # 'LingMessCoref' or "FCoref"
        "device": "cpu",                # Use "cuda" if a GPU is available
        "model_path": model_path  # Path to the local model
    }
)

#将该文本中所有的代词替换
def coreference_replace(text):
    doc = nlp(
    text, 
    component_cfg={"fastcoref": {'resolve_text': True}}
    )
    return doc._.resolved_text 

#将该本文中的出现在other_name中的词替换为aspect
def replace_words_with_aspect(text, other_name_list, aspect):#other_name_list:[""]
    """
    将 text 中所有出现在 word_list 中的词替换为 aspect。

    :param text: 输入的文本（字符串）
    :param word_list: 需要替换的词列表（字符串列表）
    :param aspect: 用于替换的词（字符串）
    :return: 替换后的文本
    """
    for word in other_name_list:
        text = text.replace(word, aspect)
    return text

text="It was Manchester City, the great domestic power of the modern era, who found themselves trapped in this perfect storm, surely tossed and blown out of contention for a fifth successive Premier League title."
text6="Virtually all of them approved of his cabinet picks, hailing them as much-needed disruptors to what they see as a corrupt establishment."
#print(coreference_replace(text7))

