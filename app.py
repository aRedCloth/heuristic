from flask import Flask, request
from flask_cors import CORS, cross_origin
from textprocess.textProcessor import get_text_byID, update_all_token_sets,get_all_token_sets,replace_all_token_sets
from textprocess.textSpanizer import phrase_add_to_patterns
from server.LabelFunction.LF_by_form import spanize_and_label_text_by_sentence, sentiment_classification_for_text,sentiment_classification_for_sentence_strict,get_selected_text_form,load_labeling_functions,update_labeling_functions
from server.LabelFunction.selfLabelFunc import queryLabelFunc
from server.embeddings.model_finetune import al_unlabel_sample,modelFineTuning
from server.embeddings.huggingfaceModel import bertEmbedding
from server.utils.helper import get_correct_path,ReturnWarningInfo, ReturnSuccessInfo, Dict2Json, Json2Dict
from server.llm.llmModel import chatllm
app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False
cors = CORS(app)
app.config['CORS_HEADER'] = 'Content-Type'


#根据index返回切割成单词和短语列表（并贴上标签）后的文本
@app.route("/textspan", methods=['POST'])
@cross_origin()
def handletext():
    data = request.get_json()
    args = data.get("args")
    idx = args["text_idx"]
    
    text=get_text_byID(idx)#根据id
    whole_span_list=spanize_and_label_text_by_sentence(text)
    ans = {"text":text,"whole_span_list":whole_span_list}
    return ReturnSuccessInfo(data=ans)

#接收前端的tokensets并更新后端的all-token-sets以及其中短语模式匹配
@app.route("/updateTokensets",methods=['POST'])
@cross_origin()
def handleUpdateTokensets():          
    data = request.get_json()
    args = data.get("args")
    tokensets = args["tokensets"]#dictionary:{"label":[token1...tokenN]}
    update_all_token_sets(tokensets)#存储到tokensets
    phrase_add_to_patterns(tokensets)#其中的短语存储到模式匹配，以便可以识别文本中的短语
    ans = {"data": "updated success!"}
    return ReturnSuccessInfo(data=ans)


#像前端返回alltokensets
@app.route("/getalltokenset", methods=["GET"])
@cross_origin()
def getAllTokenset():
    all_token_sets=get_all_token_sets()
    ans = {"data": all_token_sets}
    return ReturnSuccessInfo(data=ans)

#查询当前创立的标价函数应用于当前训练集的效果
@app.route("/querylabelfunc", methods=["POST"])
@cross_origin()
def getLabelFunction():
    data = request.get_json()
    args = data.get("args")
    ans = {"lfs_info":queryLabelFunc(args=args["lf_config"]) }
    #print(ans)
    return ReturnSuccessInfo(data=ans)

#模型微调后更新embedding以及主动学习采样点
@app.route("/finetuning", methods=["POST"])
@cross_origin()
def getFineTuningEmbedding():
    data = request.get_json()
    args = data.get("args")
    #print("app/finetuning:", args)
    '''包含{}
    1所有第三方标记函数的lfs_info:[多个{conflicts coverage coverdots&对应label， isshow, label种类（pos和neg）,name,overlaps }]
    2自定义函数集self_lfs:[name,copyname,label(这里指标记操作),conditon,coverage coverdots&对应label，isshow,tokens:[key,value,type],aspect,direction,range],
    3专家注释export_anno:{pos:[item_idx],neg:[idx],ignore,task} 
    '''
    MODEL_NAME = "roberta-base"

    dr_embedding, aclsamples, aclfsvotesamples, model_info = modelFineTuning(args)

    # 保存主动学习模型采样的节点
    ans = { "data": aclsamples }
    Dict2Json(get_correct_path("./results/"+ MODEL_NAME +"-activelearning-samples.json"), ans)

    # 保存主动学习标记函数投票不一致的节点
    ans = { "data": aclfsvotesamples }
    Dict2Json(get_correct_path("./results/"+ MODEL_NAME +"-activelearning-labelfunctionvote-samples.json"), ans)

    # 保存模型信息
    ans = { "data": model_info }
    Dict2Json(get_correct_path("./results/"+ MODEL_NAME +"-model-info.json"), ans)

    # 保存样本点的坐标
    ans = { "data": dr_embedding }
    Dict2Json(get_correct_path("./results/"+ MODEL_NAME +"-finetuning-embedding.json"), ans)

    return ReturnSuccessInfo(data=ans)


#切换模型时，更新embedding
@app.route("/embedding", methods=["POST"])
@cross_origin()
def getEmbeddingByConfig():
    data = request.get_json()
    args = data.get("args") 
    MODEL_NAME = "roberta-base"
    embeddings=bertEmbedding(args=args)
    ans = { "data":  embeddings}
    Dict2Json(get_correct_path("./results/"+ MODEL_NAME +"-embedding.json"), ans)
    return ReturnSuccessInfo(data=ans)

#查询模型信息
@app.route("/querymodelinfo", methods=["GET"])
@cross_origin()
def queryModelInfo():
    MODEL_NAME = "roberta-base"
    ans = Json2Dict(get_correct_path("./results/"+ MODEL_NAME + "-model-info.json"))
    return ReturnSuccessInfo(data=ans)


#主动学习采样
# 获取主动学习模型不确定性采样样本点 微调后生成并存储本地
@app.route("/queryalsamples", methods=["GET"])
@cross_origin()
def getActiveLearingSamples():
    MODEL_NAME = "roberta-base"
    ans = Json2Dict(get_correct_path("./results/"+ MODEL_NAME +"-activelearning-samples.json"))
    return ReturnSuccessInfo(data=ans)

# 获取主动学习标记函数不确定性采样样本点 微调后生成并存储本地
@app.route("/queryallfvotesamples", methods=["GET"])
@cross_origin()
def getActiveLearingLabelFunctionVoteUncertainSamples():
    MODEL_NAME = "roberta-base"
    ans = Json2Dict(get_correct_path("./results/"+ MODEL_NAME + "-activelearning-labelfunctionvote-samples.json"))
    return ReturnSuccessInfo(data=ans)

# 获取未被标记的 样本点
@app.route("/queryalunlabelamples", methods=["GET"])
@cross_origin()
def getActiveLearingUnlabelSamples():
    ans = {"data": al_unlabel_sample()}
    return ReturnSuccessInfo(data=ans)


# 用户问答LLM
@app.route("/chatwithllm", methods=["POST"])
@cross_origin()
# {"task": "label_and_explain/tokensets_extension/label_func_recommend", "aspect":"","points": [], 
    # "tokensets":[],"content": "", "system_task": Task-SC / Task-ABSA}
def getChatWithLLM():
    data = request.get_json() 
    args = data.get("args")

    print("its chatwithllm", args)
    ans = { "data": chatllm(args=args)}
    return ReturnSuccessInfo(data=ans)


#根据传来的form以及label,以label:form的形式把这个标记函数存入后端
@app.route("/labelfuncSubmit", methods=['POST'])
@cross_origin()
def handlelabelfuncSubmit():
    data = request.get_json() 
    args = data.get("args")
    
    form = args["form"]
    label = args["label"]
    label_func={"label":args["label"], "form":args["form"]}
    update_labeling_functions(label_func)
    return ReturnSuccessInfo()



#根据传来的(扫描选中的)文本返回切割成单词和短语列表（并贴上标签）后的文本
@app.route("/sentencespan", methods=['POST'])
@cross_origin()
def handleSentence():
    data = request.get_json() 
    args = data.get("args")
    text = args["text"]
    whole_span_list=spanize_and_label_text_by_sentence(text)
    #print(args)
    form=get_selected_text_form(whole_span_list)#得到选中句子的句型
    all_labeling_functions=load_labeling_functions()#得到所有标记函数
    sentiment_label=sentiment_classification_for_sentence_strict(form,all_labeling_functions)#将该句型与所有标记函数进行匹配，得到最合适的标签
    ans = {"text":text,"whole_span_list":whole_span_list,"form":form,"label":sentiment_label}
    return ReturnSuccessInfo(data=ans)

if __name__ == '__main__':
    print('run 0.0.0.0:14449')
    app.run(host='0.0.0.0', port=14449)
    #7890