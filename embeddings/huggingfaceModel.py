
from transformers import pipeline, AutoTokenizer, AutoModel, AutoConfig, AutoModelForSequenceClassification, T5Tokenizer, T5Model
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import numpy as np
from server.utils.helper import get_correct_path
import server.utils.config as config
dataset_path=get_correct_path("../"+config.train_dataset_path)

import os
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:1080'
os.environ["HTTP_PROXY"] = 'http://127.0.0.1:1080'

def df_filter(df):
    return df[df['label'] != 2]

def dataloader():
    df = pd.read_csv(dataset_path)
    # df = df.drop(columns=['prediction', 'prompt', 'pre_label', 'bool'])
    # df = df_filter(df)
    #df['label'] = df.apply(lambda x : 'UNLABEL', axis = 1)
    # df['t_label'] = df.apply(lambda x : 'Negative' if x.target == 0 else 'Positive', axis = 1)
    return df, df['text'].to_list()

def PCATransformation(X):
    pca = PCA(n_components=2, random_state=42)
    X_r = pca.fit(X).transform(X)
    print(
        "explained variance ratio (first two components): %s"
        % str(pca.explained_variance_ratio_)
    )
    #explained variance ratio (first two components): [0.99889964 0.00110037]
    return X_r

def TSNETransformation(X):
    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(X)
    return X_tsne

def bertEmbedding(args):
    DR_ALGO = args["dralgo"]
    #DR_ALGO = "PCA"
    df, raw_inputs = dataloader()

    #获得所有text的嵌入
    embedding = gain_embedding_batch(raw_inputs)

    MODEL_NAME = "roberta-base"
    torch.save(embedding, get_correct_path("../results/" + MODEL_NAME + "-embedding.pt"))

    X = embedding
    X = X.reshape(X.shape[0], -1) # flatten

    dr = []
    if DR_ALGO == "PCA":
        dr = PCATransformation(X)
    elif DR_ALGO == "TSNE":
        dr = TSNETransformation(X)
    #标签是直接读取的train_data_set中的标签
    labels = df["label"].to_list()

    #print("TEST IF THE LABELS OF NODE CORRECT!:",test_embedding(labels))

    dr_embedding = []
    for idx, ele in enumerate(dr):
        dr_embedding.append({
            "id": idx,
            "x": str(ele[0]),
            "y": str(ele[1]),
            "label": labels[idx]
        })
    #dr_embeddings=[{'id','x','y','label'}]
    
    return dr_embedding

#使用PLM获得embeddings
def gain_embedding_batch(sentences):
    #wa 和modelFineTuning中代码一样
    MODEL_NAME = "roberta-base"
    #MODEL_PATH = "../models/" + MODEL_NAME + "/"
    modelpath= "../models/" + MODEL_NAME + "/"
    MODEL_PATH=get_correct_path(modelpath)


    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModel.from_pretrained(MODEL_PATH)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 将句子分批次处理以减少内存使用
    batch_size = 2
    embeddings = []

    model.eval()
    with torch.no_grad():
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i+batch_size]
            inputs = tokenizer.batch_encode_plus(batch, padding='max_length', max_length=256, truncation=True, return_tensors="pt").to(device)
            outputs = model(**inputs)
            # 获取 [CLS] 标记对应的嵌入
            cls_embeddings = outputs.last_hidden_state
            embeddings.append(cls_embeddings)

    # 将嵌入拼接成一个张量
    embeddings_tensor = torch.cat(embeddings, dim=0)
    #print(embeddings_tensor.shape)
    return embeddings_tensor.cpu().numpy()



def test_embedding(df_labels=[]):
    # 读取训练集数据
    def get_label_of_train():
        # 读取CSV文件
        file_path = '../'+config.train_dataset_path # 替换为你的CSV文件路径
        current_dir = os.path.dirname(os.path.abspath(__file__))  # 当前脚本所在目录
        full_path = os.path.join(current_dir, file_path)  # 拼接成完整路径

        data = pd.read_csv(full_path)

        # 提取'polarity'列
        if 'polarity' in data.columns:
            text_column = data['polarity']
            text_list = text_column.tolist()
            return text_list
        else:
            return False
        

    def calculate_accuracy(list1, list2):
        # 确保两个列表长度一致，否则仅比较最小长度部分
        min_len = min(len(list1), len(list2))
        
        # 统计按序相同的元素数量
        correct_count = sum(1 for i in range(min_len) if list1[i] == list2[i])
        
        # 计算正确率
        accuracy = correct_count / min_len if min_len > 0 else 0.0
        return accuracy
    labels_train=get_label_of_train()
    acc=calculate_accuracy(labels_train,df_labels)
    return acc
