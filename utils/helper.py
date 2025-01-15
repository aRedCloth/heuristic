import json
from flask import jsonify
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict
import pandas as pd
import os
import inspect
import server.utils.config as config


def get_correct_path(target_path=""):
    # 获取调用者的文件路径
    caller_file = inspect.stack()[1].filename
    caller_dir = os.path.dirname(os.path.abspath(caller_file))
    final_path = os.path.join(caller_dir, target_path)
    return final_path


def ReturnInfo(code, type, msg, data) -> dict:
    return jsonify({
        "code": code,
        "type": type,
        "msg": msg,
        "data": data
    })

def ReturnSuccessInfo(code=0, type="success", msg="success", data=None) -> dict:
    return ReturnInfo(code, type, msg, data)

def ReturnWarningInfo(code=20010, type="warning", msg="warning", data=None) -> dict:
    return ReturnInfo(code, type, msg, data)
#absa_log
def Dict2Json(outfile, data):

    with open(outfile,'w') as f:
        json.dump(data, f)

def Json2Dict(file):
    with open(file, 'r') as f:
        ans = json.load(f)
    return ans
    
'''wa输出形式如下：
{
    "data": ["not", "nobody", "nothing", "no one"],
    "value": "NEGATION"
}
'''
def df_shuffle(df, test_size=0.2):
    df = df.sample(frac=1).reset_index(drop=True)
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=42)
    return train_df, test_df


def createDatasetDict(df):
    train_df, test_df = df_shuffle(df)
    train_data = pd.DataFrame({
        'text': train_df['text'].to_list(),
        #'label': [config.POSITIVE if ele =="positive" else config.NEGATIVE for ele in train_df['label'].to_list()]
        'label': [config.POSITIVE if ele =="positive" else (config.NEGATIVE if ele=="negative" else config.NEUTRAL) for ele in train_df['label'].to_list()]
    })
    test_data = pd.DataFrame({
        'text': test_df['text'].to_list(),
        #'label': [config.POSITIVE if ele =="positive" else config.NEGATIVE for ele in test_df['label'].to_list()]
        'label': [config.POSITIVE if ele =="positive" else (config.NEGATIVE if ele=="negative" else config.NEUTRAL) for ele in test_df['label'].to_list()]
    })
    data = {
        'train': Dataset.from_pandas(train_data),
        'test': Dataset.from_pandas(test_data)
    }
    dataset_dict = DatasetDict(data)
    dataset_dict = dataset_dict.map(lambda example: {'text': example['text'], 'label': example['label']})
    return dataset_dict

#从数据集中读取训练集的文本和标签，以list形式返回
def gain_test_dataset():
    df = pd.read_csv(get_correct_path('../'+config.train_dataset_path))                                                                           
    label_list = df["polarity"].to_list()
    #label_list = [1 if ele==4 or ele=="positive" else 0 for ele in label_list]

    label_list = [1 if ele == "positive" else (0 if ele == "negative" else 2) for ele in label_list]
    #print("test gained label_list", label_list)
    text_list = df["text"].to_list()
    return text_list, label_list

def label_string2int(label):#把string类型的label转换为适宜snorkel标记函数的int型
    if(label=="positive"):
        return config.POSITIVE
    elif(label=="negative"):
        return config.NEGATIVE
    else:
        return config.NEUTRAL

def extract_activelearning_data():
    df = pd.read_csv(get_correct_path('../'+config.train_dataset_path))                                                                           
    text_list = df["text"].to_list()
    MODEL_NAME="roberta-base"
    points = Json2Dict(get_correct_path("../results/"+ MODEL_NAME +"-activelearning-samples.json"))["data"]
    ac_text_list=[]
    for point in points:
        ac_text_list.append(text_list[point])
    return ac_text_list

