import random
from server.LabelFunction.LF_by_svo_extract import extract_svo,label_with_svo_method,filtered_svo_with_aspect
from server.LabelFunction.LF_by_token import label_with_token_match_method,label_with_window_analysis_method,label_with_sc_method
from server.LabelFunction.co_reference import coreference_replace
from server.textprocess.textProcessor import get_all_token_sets,get_custom_token_sets
from snorkel.labeling import LabelingFunction, labeling_function, PandasLFApplier, LFAnalysis, filter_unlabeled_dataframe
from snorkel.labeling.model import MajorityLabelVoter, LabelModel
import pandas as pd
import numpy as np
import os

from server.utils.helper import  get_correct_path
import server.utils.config as config
import string
from server.LabelFunction.co_reference import coreference_replace,replace_words_with_aspect
from server.LabelFunction.LF_helper import match_label_for_form_with_rules,get_label_for_form,rules,sentiment_labels
file_path = '../'+config.train_dataset_path  # 替换为你的CSV文件路径

current_dir = os.path.dirname(os.path.abspath(__file__))  # 当前脚本所在目录
dataset_path = os.path.join(current_dir, file_path)  # 拼接成完整路径


# 根据用户传入的参数构建标记函数；向前端返回标记函数信息lfsinfo
def queryLabelFunc(args):
    lf = [make_absa_lf(args)]
    df_train, L_train, lfsinfo = computeLFS(lfs= lf, isThird=False)
    #print(lfsinfo[0])
    return lfsinfo[0]
    # lfsinfo [{'name': 'LF_positive-token-set_', 'label': [], 'coverage': '10.0%', 'overlaps': '0.0%', 'conflicts': '0.0%', 'coverdots': [], 'ishow': 'view'} ]

#根据前端传来的参数创建标记函数
def make_absa_lf(args):
    #print(args)
    lf_name=args["func_name"]
    aspect = args["aspect"]
    pronoun_replace_flag=args["pronoun_replace"] #'True'/'False'
    other_name_list = args["other_name"]#["name"]
    label_method=args["label_method"]
    token_match_config=args["token_match_config"]#{"direction":forward,backward,near}
    structure_match_config=args["structure_match_config"]
    window_analysis_config=args["window_analysis_config"]#{"window_size":''}
    label_list=args["selected_tokensets"]#用户选择的tokensets(前端只传入标签)
    custom_token_sets=get_custom_token_sets(label_list)#用户选择的tokensets
    
    sentiment_labels=args["sentiment_labels"]
    rules=args["rules"]

    # 在将文本传入标记函数前对齐进行动态的预处理，比如代词替换
    def preprocess_text(text):
        """根据参数对文本进行预处理"""
        # 转小写（通用步骤）
        text = text.lower()
        # 替换代词
        if len(other_name_list)>1 or (len(other_name_list)==1 and other_name_list[0]!=''): #[{'name': ''}]表示other_name_list为空
            text =replace_words_with_aspect(text, other_name_list, aspect)
        if pronoun_replace_flag=="True":
            text = coreference_replace(text) 
        return text

    #标记函数模板，用来构建适用于snorkel的标记函数,便于处理结点x的文本后传入真正用于逻辑处理的标记函数中
    def lf_svo_in_snorkel_template(x,aspect,alltokensets,sentiment_labels,rules):
        preprocessed_text = preprocess_text(x.text)
        return label_with_svo_method(preprocessed_text,aspect,alltokensets,sentiment_labels,rules)
    
    def lf_token_match_in_snorkel_template(x,aspect,direction,alltokensets,sentiment_labels,rules):
        preprocessed_text = preprocess_text(x.text)
        return label_with_token_match_method(preprocessed_text,aspect,direction,alltokensets,sentiment_labels,rules)

    def lf_sc_in_snorkel_template(x,aspect,alltokensets,sentiment_labels,rules):
        preprocessed_text = preprocess_text(x.text)
        return label_with_sc_method(preprocessed_text,aspect,alltokensets,sentiment_labels,rules)
    
    def lf_window_in_snorkel_template(x, aspect, alltokensets, window_size,sentiment_labels,rules):
        preprocessed_text = preprocess_text(x.text)
        return label_with_window_analysis_method(preprocessed_text, aspect, alltokensets, window_size,sentiment_labels,rules)

    #1:svo labeling function
    if(label_method=='structure-match'):
        return LabelingFunction(
        name = lf_name + aspect + "_" + generate_random_string(10),
        f=lf_svo_in_snorkel_template,
        resources={"aspect": aspect, "alltokensets":custom_token_sets,"sentiment_labels":sentiment_labels,"rules":rules}  # 注入 aspect 作为资源
    )

    #2:sc labeling function
    if(label_method=='full-text-match'):
        return LabelingFunction(
        name = lf_name + aspect + "_" + generate_random_string(10),
        f=lf_sc_in_snorkel_template,
        resources={"aspect": aspect,"alltokensets":custom_token_sets,"sentiment_labels":sentiment_labels,"rules":rules}  # 注入 aspect 作为资源
    )

    #3:token_match labeling function
    elif(label_method=='token-match'):
        return LabelingFunction(
        name = lf_name + aspect + "_" + generate_random_string(10),
        f=lf_token_match_in_snorkel_template,
        resources={"aspect": aspect,"direction":token_match_config["direction"],"alltokensets":custom_token_sets,"sentiment_labels":sentiment_labels,"rules":rules}  # 注入 aspect 作为资源
    )
    
    #4:window_analysis labeling functino
    elif(label_method=='window-analysis-match'):
        window_size = int(window_analysis_config.get("window_size", 6))  # 默认值为 6
        return LabelingFunction(
            name=lf_name + aspect + "_" + generate_random_string(10),
            f=lf_window_in_snorkel_template,
            resources={"aspect": aspect, "alltokensets": custom_token_sets, "window_size": window_size,"sentiment_labels":sentiment_labels,"rules":rules}  # 注入 aspect 和 window_size 作为资源
        )


#将当前的标记函数应用于当前数据集，得到原始数据集df_train, 各数据被各函数标注的矩阵L_train，以及各标记函数的信息lfsinfo
def computeLFS(lfs = [], isThird=True):
    df_train = pd.read_csv(dataset_path)#restaurants-train.csv (id,text,term,polarity,label)
    applier = PandasLFApplier(lfs=lfs)#snorkel
    L_train = applier.apply(df=df_train)
    snorkel_analysis = LFAnalysis(L=L_train, lfs=lfs).lf_summary()#包含j(第j个函数),  Polarity,conflict,overlap,coverage
    lfsinfo = lfs_df_to_dict_list(snorkel_analysis, L_train, isThird=isThird)
    #print("lfs_df_to_dict_list",lfsinfo)
    return df_train, L_train, lfsinfo

#获取各标记函数的信息'name' 'label' 'coverage' 'overlaps' 'conflicts' 'coverdots' 'coverlabels': ['Positive', 'Negative'], 'ishow': 'view'/'hide'
def lfs_df_to_dict_list(snorkel_analysis, L_train, isThird=True):
    #cov = (L_train != config.NEUTRAL)
    cov=L_train
    dict_list = []
    names = snorkel_analysis.index.tolist()
    tmp = snorkel_analysis.to_dict(orient='records') # df 每行 转 字典 每一行代表了一个标记函数的所有信息
    for id, ele in enumerate(tmp):
        ele["name"] = names[id].split("_")[0].upper() if isThird else names[id]
        ele["j"] = int(ele["j"])
        ele["label"] = ["negative" if ei == 0 else (1 if ei=="positive"  else "neutral")for ei in ele["Polarity"]]
        
        ele["coverage"] = str(round(ele["Coverage"] * 100, 1)) + "%"
        ele["overlaps"] = str(round(ele["Overlaps"] * 100, 1)) + "%"
        ele["conflicts"] = str(round(ele["Conflicts"] * 100, 1)) + "%"
        # 查寻每个标记函数标记了哪些数据点
        ele["coverdots"] = np.where(cov[:, ele["j"]])[0].tolist()

        # print(ele["coverdots"])
        # print()
        # 输出L_train矩阵某一列，也就是第j个标记函数的标记结果
        # print(L_train[:, ele["j"]])

        # 查询每个数据点的标签
        ele["coverlabels"] = []
        for ei in L_train[:, ele["j"]]:
            if ei == 1:
                ele["coverlabels"].append("positive")
            elif ei == 0:
                ele["coverlabels"].append("negative")
            else:
                continue
        
        # print(len(ele["coverlabels"]))
        # print(len(ele["coverdots"]))
        # print()
        # 标记每个标记函数是否在前端显示
        ele["ishow"] = "view" # ["view", "hide"]
        
        del ele['j']
        del ele['Polarity']
        del ele['Coverage']
        del ele['Overlaps']
        del ele['Conflicts']
        dict_list.append(ele)
    return dict_list

#使用snorkel模型整合L_train，获得对每个样本的最终预测标签并写回dataset.csv，最后返回过滤掉未标注样本后的df_train和probs_train
def snorkelPredLabels(df_train, L_train, expert_anno):

    def calculate_accuracy(df_train):
        # 比较 'polarity' 和 'label' 列是否相同
        correct_predictions = (df_train["polarity"] == df_train["label"]).sum()

        # 计算准确率
        accuracy = correct_predictions / len(df_train)

        return accuracy

    # 预测
    label_model = LabelModel(cardinality=3, verbose=True)
    label_model.fit(L_train=L_train, n_epochs=600, log_freq=100, seed=123)
    # 写文件
    # 这里是标记函数的预测值
    label_model_preds = label_model.predict(L = L_train)
    df_train["label"] = ["positive" if yi==config.POSITIVE else ("negative" if yi==config.NEGATIVE else "neutral" )for yi in label_model_preds]
    #print("labelModel预测：", label_model_preds)


    # 这里是专家标记值，修改积极的消极的
    pos_rows = expert_anno["positive"]
    df_train.loc[pos_rows, "label"] = "positive"
    L_train[pos_rows] = config.POSITIVE 

    neg_rows = expert_anno["negative"]
    df_train.loc[neg_rows, "label"] = "negative"
    L_train[neg_rows] = config.NEGATIVE

    ntrl_rows = expert_anno["neutral"]
    L_train[ntrl_rows] = config.NEUTRAL
    df_train.to_csv(dataset_path, index=False, encoding='utf-8')
    print("snorkel标记函数准确率：",calculate_accuracy(df_train))
    
    probs_train = label_model.predict_proba(L=L_train)#proba概率分布
    # 过滤

    df_train_filtered, probs_train_filtered = filter_unlabeled_dataframe(
        X=df_train, y=probs_train, L=L_train
    )
    
    return df_train_filtered, probs_train_filtered
    
    #return df_train,probs_train
    '''''' 

def majorityPredLabels(L_train):
    majority_model = MajorityLabelVoter()
    preds_train = majority_model.predict(L=L_train)
    print(preds_train)
    return preds_train


# 读取测试集标签
def get_label_of_test():
    # 读取CSV文件
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

def generate_random_string(length):
    # 生成大小写字母的所有字符
    letters = string.ascii_letters
    # 生成随机字符串
    random_string = ''.join(random.choice(letters) for _ in range(length))
    return random_string

# 计算准确率
def calculate_accuracy(list1, list2):
    # 确保两个列表长度一致，否则仅比较最小长度部分
    min_len = min(len(list1), len(list2))
    
    # 统计按序相同的元素数量
    correct_count = sum(1 for i in range(min_len) if list1[i] == list2[i])
    
    # 计算正确率
    accuracy = correct_count / min_len if min_len > 0 else 0.0
    return accuracy

#检查对于一个文本，各个标记函数的标记状况
def check_labeling_results(text,aspect,alltokensets,sentiment_labels,rules):
    svo_label=label_with_svo_method(text,aspect,alltokensets,sentiment_labels,rules)
    token_match=label_with_token_match_method(text,aspect,"near",alltokensets,sentiment_labels,rules)
    window=label_with_window_analysis_method(text,aspect,alltokensets,7,sentiment_labels,rules)
    sc=label_with_sc_method(text,aspect,alltokensets,sentiment_labels,rules)
    print("svo_label",svo_label)
    print("tokenmatch_label",token_match)
    print("window_label",window)
    print("sc_label",sc)
    

if __name__=='__main__':

    rules= {'positive': [['goodfood'], ['negation', 'negative'],['negation', 'badfood']], 
            'negative': [ ['negation', 'goodfood'],['negation', 'positive'], ['badfood']]}, 
    sentiment_labels=['positive','negative']
    aspect='food'
    alltokensets=get_all_token_sets()

    while(1):
        text=input("Enter the text:  ")
        check_labeling_results(text.lower(),aspect,alltokensets,sentiment_labels,rules)


    token_match_config={'func_name': 'token_match', 'aspect': 'food', 'pronoun_replace': 'False', 'other_name': ['seafood','foods'], 
                        'sentiment_labels': ['positive', 'negative','neutral'], 
                        'rules':  {'positive': [['goodfood'], ['negation', 'negative'],['negation', 'badfood']], 
                                    'negative': [ ['negation', 'goodfood'],['negation', 'positive'], ['badfood']]},
                'selected_tokensets': ['positive', 'negation', 'negative','goodfood','badfood','contrast','neutral'], 
                        'label_method': 'token-match', 'token_match_config': {'direction': 'near'}, 'structure_match_config': {'clause_complement': ''}, 'window_analysis_config': {'window_size': 1}}     

    svo_config={'func_name': 'svo', 'aspect': 'food', 'pronoun_replace': 'False', 'other_name': ['seafood','foods'], 
                'sentiment_labels': ['positive', 'negative','neutral'], 
                'rules':  {'positive': [['goodfood'], ['negation', 'negative'],['negation', 'badfood']], 
                            'negative': [ ['negation', 'goodfood'],['negation', 'positive'], ['badfood']]},
                'selected_tokensets': ['positive', 'negation', 'negative','goodfood','badfood','contrast','neutral'],  
                'label_method': 'structure-match', 'token_match_config': {'direction': 'near'}, 
                'structure_match_config': {'clause_complement': ''}, 'window_analysis_config': {'window_size': 1}}  


    window_config={'func_name': 'windo5w', 'aspect': 'food', 'pronoun_replace': 'False', 'other_name': ['seafood','foods'], 
                    'sentiment_labels': ['positive', 'negative','neutral'], 
                    'rules':  {'positive': [['goodfood'], ['negation', 'negative'],['negation', 'badfood']], 
                                'negative': [ ['negation', 'goodfood'],['negation', 'positive'], ['badfood']]},
                'selected_tokensets': ['positive', 'negation', 'negative','goodfood','badfood','contrast','neutral'],  
                    'label_method': 'window-analysis-match', 'token_match_config': {'direction': 'near'}, 'structure_match_config': {'clause_complement': ''}, 'window_analysis_config': {'window_size': '5'}}


    sc_config={'func_name': 'sc', 'aspect': 'food', 'pronoun_replace': 'False', 'other_name': ['seafood','foods'], 
                'sentiment_labels': ['positive', 'negative','neutral'], 
                'rules':  {'positive': [['goodfood'], ['negation', 'negative'],['negation', 'badfood']], 
                            'negative': [ ['negation', 'goodfood'],['negation', 'positive'], ['badfood']]},
                'selected_tokensets': ['positive', 'negation', 'negative','goodfood','badfood','contrast','neutral'],  
                'label_method': 'full-text-match','token_match_config': {'direction': 'near'}, 'structure_match_config': {'clause_complement': ''}, 'window_analysis_config': {'window_size': '15'}}
    
    # 假设标注函数已经定义
    token_match =make_absa_lf(token_match_config)
    svo=make_absa_lf(svo_config)
    window_match = make_absa_lf(window_config)
    sc = make_absa_lf(sc_config)

    lfs = [svo,token_match, window_match,sc]

    # 读取训练集数据
    file2_path = '../datasets/restaurants-train.csv'
    current_dir = os.path.dirname(os.path.abspath(__file__))  # 当前脚本所在目录
    dataset2_path = os.path.join(current_dir, file2_path)  # 拼接成完整路径
    df_train = pd.read_csv(dataset2_path)


    # 应用标注函数
    applier = PandasLFApplier(lfs=lfs)
    L_train = applier.apply(df=df_train)
    print("check my L_train",L_train[190:210])

    expert={'positive':[],"negative":[],"neutral":[]}
    df_fil, pred_fil= snorkelPredLabels(df_train, L_train,expert)


    # 训练模型
    label_model = LabelModel(cardinality=3, verbose=True)
    label_model.fit(L_train=L_train, n_epochs=300, log_freq=100,seed=666)
    label_model_preds = label_model.predict(L=L_train)

    # 预测标签
    pred_labels = []
    for item in label_model_preds:
        if item == 1:
            pred_labels.append("positive")
        elif item == 0:
            pred_labels.append("negative")
        else:
            pred_labels.append("neutral")

    # 获取测试标签
    test_label_list = get_label_of_test()
    if not test_label_list:
        print("Test labels not found or invalid")
    else:
        print("test_pred_labels",pred_labels[:20])
        print("test true labels",test_label_list[:20])
        print("accuracy:", calculate_accuracy(pred_labels, test_label_list))
    # 分析 Labeling Functions
    analysis = LFAnalysis(L=L_train, lfs=lfs).lf_summary()#包含j(第j个函数),  Polarity,conflict,overlap,coverage
    lfsinfo = lfs_df_to_dict_list(analysis, L_train, isThird=True)
    #df_train_filtered, probs_train_filtered=snorkelPredLabels(df_train, L_train, [])

    for lf in lfsinfo:
        print(lf["name"], " acc:", calculate_accuracy(lf["coverlabels"], test_label_list))

