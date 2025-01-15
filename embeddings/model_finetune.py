import importlib
import os
from server.LabelFunction.selfLabelFunc import computeLFS, snorkelPredLabels, majorityPredLabels
from transformers import AutoTokenizer, AutoModel, Trainer, RobertaTokenizer, RobertaForSequenceClassification, TrainingArguments, DataCollatorWithPadding
from server.utils.helper import createDatasetDict
from server.embeddings.huggingfaceModel import PCATransformation
from server.LabelFunction.selfLabelFunc import make_absa_lf
import torch
import torch.nn.functional as F
import numpy as np
from collections import Counter
import math
from server.utils.helper import get_correct_path
import pandas as pd
import server.utils.config as config

train_path=get_correct_path("../"+config.train_dataset_path)
test_path=get_correct_path("../"+config.test_dataset_path)

def modelFineTuning(args):
    
    '''args 包含lfs_info,一个列表，表中每一项是一个标记函数的配置信息和性能表现
    
    lfs_info:[ {
                config:{lf的配置信息},func_name,aspect,pronoun_replace,other_name: [{ key: Date.now(), name: '' }],   label_method,condition，
                                    token_match_config:{window_size:{left:'' ,right:''}},
                                    structure_match_config:{clause_complement:''},

                perform:{lf的性能表现}conflicts coverage coverdots&对应label， ishow, label种类（pos和neg）,name,overlaps 
                } ]

    '''
    # 第一步 通过标记函数信息找到对应的标记函数；如果这里面的标记函数不是第三方标记函数而是用户自定义的标记函数，那就需要进行解析用户自定义的标记函数
    
    lfs = []
    lfs_name = []


    # 里获取用户自定义的标记函数
    for ele in args["lfs_info"]:
        # 过滤前端展示的标记函数
        if ele["perform"]["ishow"]=="view":
            func_name = ele["config"]["func_name"]
            lfs_name.append(func_name)
            func = make_absa_lf(ele["config"])
            lfs.append(func)

    print("in model-finetune's lfs:",lfs_name)

    # 这里获取用户标记的数据
    expert_anno = args["expert_anno"]
    print("in model-finetune's expertAnno:",expert_anno)

    # 第二步 通过lfs计算L_train
    df_train, L_train, lfsinfo = computeLFS(lfs=lfs)
    # 第三步 通过L_train 与 snorkel 计算伪标签
    # 第四步 过滤未标记的数据集， 注意：这里同时也要加上专家的标注结果{positive:[], negative:[], ignore:[]}，需要注意的是，ignore是直接忽略节点
    df_train_filtered, probs_train_filtered = snorkelPredLabels(df_train=df_train, L_train=L_train, expert_anno=expert_anno)
    # 第五步 构建训练数据集、测试数据集
    raw_datasets = createDatasetDict(df_train_filtered)
    print(raw_datasets)


    # 第六步 微调模型、保存模型
    MODEL_NAME = "roberta-base"
    #MODEL_PATH = "../models/" + MODEL_NAME + "/"
    modelpath= "../models/" + MODEL_NAME + "/"
    MODEL_PATH=get_correct_path(modelpath)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = RobertaForSequenceClassification.from_pretrained(MODEL_PATH, num_labels=config.CLASSIFICATION_DIMENSION)
    tokenizer = RobertaTokenizer.from_pretrained(MODEL_PATH, max_length=256)

    def tokenize_function(example):
        return tokenizer(example["text"], truncation=True)
    
    tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    training_args = TrainingArguments(
        output_dir = get_correct_path("../models/test-trainer"),
        learning_rate= 1e-05,
        num_train_epochs=5,
        save_steps= 50)
    trainer = Trainer(
        model,
        training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    trainer.train()
    trainer.evaluate()
    # 保存训练后的模型
    model.save_pretrained(get_correct_path('../models/saved_model'))
    tokenizer.save_pretrained(get_correct_path('../models/saved_model'))

    # 第七步 返回数据点二维坐标编码
    raw_inputs = df_train["text"].to_list()

    batch_size = 2# 将句子分批次处理以减少内存使用
    embeddings = []

    model.eval()

    with torch.no_grad():
        for i in range(0, len(raw_inputs), batch_size):
            batch = raw_inputs[i:i+batch_size]
            inputs = tokenizer.batch_encode_plus(batch, padding='max_length', max_length=256, truncation=True, return_tensors="pt").to(device)
            outputs = model(**inputs)
            cls_embeddings = outputs.logits
            embeddings.append(cls_embeddings)

    # 将嵌入拼接成一个张量
    embedding = torch.cat(embeddings, dim=0)

    # 主动学习采样
    # 1.底层模型 预测样本标签 概率在0.5附近
    aclsamples = active_learning_samples(embedding)
    # 2.标记函数 对样本标签的投票 投票的标签结果五五开
    aclfsvotesamples = label_function_vote_sample(L_train)
    # 3.始终未标记的数据 / 忽略的数据点

    # 计算样本点分布
    X = embedding.detach().cpu().numpy()
    X = X.reshape(X.shape[0], -1) # flatten

    dr = PCATransformation(X)
    
    labels = df_train["label"].to_list()

    #print("TEST IF THE LABELS OF NODE CORRECT!:",test_embedding(labels))
    
    dr_embedding = []
    for idx, ele in enumerate(dr):
        dr_embedding.append({
            "id": idx,
            "x": str(ele[0]),
            "y": str(ele[1]),
            "label": labels[idx]
        })
    

    # 第八步 测试集测试当前模型的准确率
    text_list, label_list = gain_test_dataset()
    pre_label_list = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(text_list), batch_size):
            batch = text_list[i:i+batch_size]
            inputs = tokenizer.batch_encode_plus(batch, padding='max_length', max_length=256, truncation=True, return_tensors="pt").to(device)
            outputs = model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=1).tolist()
            pre_label_list.extend(predictions)

    print("pre_label_list of model", pre_label_list)
    count = 0
    for e1,e2 in zip(pre_label_list, label_list):
        if e1 == e2:
            count += 1
    accuracy = count / len(label_list)
    print("准确率", accuracy)

    model_info = {
        "name": MODEL_NAME,
        "size": "477MB",
        "accuracy": accuracy,
        "train_set": len(df_train),
        "test_set": len(label_list)
    }
    return dr_embedding, aclsamples, aclfsvotesamples, model_info


def gain_test_dataset():
    df = pd.read_csv(test_path)
    label_list = df["polarity"].to_list()
    if config.CLASSIFICATION_DIMENSION==2:
        label_list = [1 if ele=="positive" else 0 for ele in label_list]
    elif config.CLASSIFICATION_DIMENSION==3:
        label_list = [1 if ele == "positive" else (0 if ele == "negative" else 2) for ele in label_list]
    print("label_list 真实标签", label_list)
    text_list = df["text"].to_list()
    return text_list, label_list



# 计算主动学习采样，目前只考虑预测概率很相近的节点（神经网络最后一层softmax可以得到两个类别的概率，概率相减的绝对值从小到大排序，获取前25个节点id），
def active_learning_samples_2_classification(embedding):

    predictions = F.softmax(embedding, dim=-1).detach().cpu().numpy()

    # 检查数组的维度和形状, 这里确保predictions 是一个nx2的二维数组
    if predictions.ndim == 2 and predictions.shape[1] == config.CLASSIFICATION_DIMENSION:
        #abs diff will be small when prob close to 0.5.
        diff_values = np.abs(np.diff(predictions, axis=1)).flatten()
        # print(diff_values)
        sorted_indices = np.argsort(diff_values)#in ascending order
        # print(sorted_indices)
        sample_len = int(len(sorted_indices) * 0.1)
        return sorted_indices[:sample_len].tolist()
    
    else:
        print(predictions)
        print("predictions 不是 nx2 二维数组")
        return []

def active_learning_samples(embedding):
    # 使用 softmax 计算预测概率
    predictions = F.softmax(embedding, dim=-1).detach().cpu().numpy()

    # 确保 predictions 是一个 n x 3 的二维数组（三分类任务）
    if predictions.ndim == 2 and predictions.shape[1] == 3:
        # 对每个样本找到预测概率最大的两个类别，并计算差异的绝对值
        sorted_probs = np.sort(predictions, axis=1)  # 按概率排序，shape 为 (n, 3)
        diff_values = np.abs(sorted_probs[:, -1] - sorted_probs[:, -2])  # 最大和次大概率的差值

        # 排序差异值并获取对应的索引（从小到大）
        sorted_indices = np.argsort(diff_values)

        # 选择最接近的 10% 的样本（你可以调整这里的比例）
        sample_len = int(len(sorted_indices) * 0.1)

        # 返回前 sample_len 个样本的索引
        return sorted_indices[:sample_len].tolist()

    else:
        # 如果 predictions 不是 nx3 的二维数组，打印并返回空列表
        print("Predictions are not a nx3 2D array")
        print("Shape of predictions:", predictions.shape)
        return []
def label_function_vote_sample(L_train):
    def entropy(data):
        counter = Counter(data)  # 统计列表中每个元素的个数
        total_count = len(data)  # 列表中元素的总个数
        entropy = 0.0
        for count in counter.values():
            probability = count / total_count  # 计算每个元素的概率
            entropy -= probability * math.log2(probability)  # 计算熵的累加值
        return entropy

    # 记录每一个样本点，各个投票函数，投票结果
    vote_diff = []

    for ele in L_train:
        vote_diff.append(entropy(ele))
    
    print(vote_diff)
    sorted_indices = np.argsort(vote_diff)
    sample_len = int(len(sorted_indices) * 0.1)

    # 返回投票不一致节点的索引数组
    return sorted_indices[-sample_len:].tolist()


def al_unlabel_sample():
    label_list = pd.read_csv(train_path)["label"].to_list()
    ans = []
    for index, value in enumerate(label_list):
        if value == "neutral":
            ans.append(index)
    return ans


def test_embedding(df_labels=[]):
    # 读取训练集数据
    def get_label_of_train():
        # 读取CSV文件
        file_path = '../datasets/restaurants-train.csv'  # 替换为你的CSV文件路径
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



