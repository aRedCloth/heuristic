
import os
from server.llm.openai.chat_openai import get_openai_completion
from server.utils.helper import Json2Dict,gain_test_dataset,get_correct_path,extract_activelearning_data
from server.textprocess.textProcessor import get_all_token_sets
from server.LabelFunction.LF_by_form import spanize_and_label_text_by_text
from datetime import datetime
import pandas as pd
import openai

def chatllm(args):
    # {"task": "label_and_explain/tokensets_extension/label_func_recommend", "aspect":"",
    # "tokensets":[],"content": "", "system_task": Task-SC / Task-ABSA}
    print(args)
    task = args["task"]
    aspect=args["aspect"]
    tokensets=args["tokensets"]
    if task == "label_and_explain":#当前文本情感分析
        content = args["content"]
        
        prompt = "This is an Aspected-Based Sentiment Analysis task. \
                I will give you a sentence and an object aspect please analyze this sentence and find the sentiment toward the object aspect.\
                the sentiment could only be one of 'positive', 'negative' or 'neutral'.\
                Please show me your reasoning process and explian why you select this sentiment for this aspect.\
                The return result should be in a JSON format: {\"label\":###the sentiment you select for this aspect###, \"reason\":### your reasoning process and explaination###}"
        prompt += "The object aspect is:"+aspect+"."
        prompt += "The sentence is: '" + content + "' "        
        prompt += "Please return JSON only, without any other comments or texts."
        
        llm_res = get_openai_completion(prompt=prompt)
        #saveLLMResopnse(llm_res)
        print("chat with gpt success!")
        return llm_res
    
    elif task == "tokensets_extension":#tokenset扩充
        #text_list, label_list=gain_test_dataset()
        ac_text_list = extract_activelearning_data()
        all_text_in_a_string=' '.join(ac_text_list[:25])
        if all_text_in_a_string == '':
            return "text length is 0!"
        else:
            prompt="I have a label list (each element in this list stands for a label category.) and sentences. Now i want you to do a word labeling task following steps below:\
                    1. Please create a dictionary format, each key is the label in this label list, and the value is an empty list for each label. \
                        For example, if the label list is ['positive','negative'], you will create a dictionary like {'positive':[],'negative':[]}\
                    2. Please read these sentences by words.\
                    3. For each word or phrase you read, please check the label list and analyze if this word (or phrase) can be annotated as this label.\
                    4. If you think the word you read now have the same meaning with a label in the label list, please add this word to the list where this label corresponds to in the dictionary.\
                    5. Once you add a word to a list, when you meet this word again, do not add it to the dictionary again. \
                    6. When you finish handling all the sentence, please return the dictionary in JSON format. The JSON should be like ### {'label1':['word1','word2'],'label2':['word3','word5']}###.\
                    Above is all the steps you need to follow, Now i will give you an success example.\
                    For example, There is a label list ['positive','negative','goodguy','badguy'] and a sentence 'I love this superhero and hate his enemy.' \
                    Within this sentence, the word 'love' can be labeled as 'positive', the word 'hate' can be labeled as 'negative', the word 'superhero' can be labeled as 'goodguy', the word 'enemy' can be labeled as 'badguy'.\
                    After analyzing this example , the return results should be {'positive':['love'],'negative':['hate'],'goodguy':['superhero'],'badguy':['enemy']}"
            prompt += "the label list is:" + str(tokensets)+'.'
            prompt += "the sentence is: "+ all_text_in_a_string +"."
            prompt +="The return result is provided in JSON format, Each different word can only been added once. "
            prompt += "Please return JSON only, without any other comments or texts. \
                Please return JSON only, without any other comments or texts."
            
            llm_res = get_openai_completion(prompt=prompt)
            #saveLLMResopnse(llm_res)
            return llm_res
        
    elif task == "label_func_recommend":    
        #text_list, label_list=gain_test_dataset()
        #all_text_in_a_string=' '.join(text_list[:10])
        ac_text_list = extract_activelearning_data()
        all_text_in_a_string=' '.join(ac_text_list[:15])
        selected_tokensets=str(tokensets)
        sentiment_label="['positive','negative','neutral']"
        text="To be completely fair, the only redeeming factor was the food, which was above average, but couldn't make up for all the other deficiencies of Teodora."
        all_tokensets=get_all_token_sets()
        sentence_dict=spanize_and_label_text_by_text(all_text_in_a_string,all_tokensets)
        func_config={'func_name': 'token_match', 'aspect': str(aspect), 'pronoun_replace': 'False', 'other_name': [], 'sentiment_labels': ['positive', 'negative','neutral'], 
                    'rules': { #一个字典，其中每一项对应了一个情感标签（例如pos,neg； 而不应该含有negation）
                            "negative":[["negative"],["negation","positive"]],
                            "positive":[ ["negation","negative"],["positive"],],
                    }, 
                    'selected_tokensets': str(tokensets), 
                    'label_method': 'token-match', 'token_match_config': {'direction': 'near'}, 'structure_match_config': {'clause_complement': ''}, 'window_analysis_config': {'window_size': 1}}     

        prompt="There is an example of function configuration template in format of dictionary:"
        prompt+=str(func_config)
        # label method and other_name
        prompt+="I am trying to configure it to contruct a Aspecet-based sentiment analysis function finding the sentiment of the aspect(namely the 'aspect' attribute in this template) in a sentence.\
                Your mission is to contruct a labeling function template for me by following the steps below and refering the form of the example i gave above\
                You can select a labeling method from 'token-match','structure-match','window-analysis-match'and 'full-text-match' for 'label_method' attribute in the template. The specific meaning of these labeling method are:\
                1. 'token-match' method firstly locates the position of 'aspect' in a sentence and find the nearest word that express a sentiment (happy or sad for instance) to the aspect. \
                    Then this method takes the sentiment of this word as the sentiment of this aspect.\
                    if you select this method, you need to configure the attribute 'direction'. 'direction' has 3 options:'near','forward','backward'.\
                    'forward' means to search nearest sentiment word forward (search for words before aspect)\
                    'backward' means to search nearest sentiment word backward (search for words behind aspect)\
                    'near' means to search nearest sentiment word both forward and backward, then takes the nearer one as the sentiment word.\
                2.'structure-match' tries to extract a subject-verb-object structure for a sentence and analyze the relationship among these 3 elements , then get a final sentiment.\
                3.'window-analysis-match'  firstly locates the position of 'aspect' and take a parameter 'window_size' as radius, \
                    then it takes the sentiment of this certain range('window_size' words before aspect and 'window_size'words behind aspect) as the sentiment of this aspect.\
                    if you select this method, please select a suitable window size and configure the attirbute 'window_size'.\
                4.'full-text-match' takes the sentiment of this sentence as the sentiment of this aspect."
    
        prompt+="Now i will give you some sentences, please analyze them and select a 'label_method' that you think is most useful for doing current ABSA task on these sentences.\
                Please also extract words that have the similar meaning to the aspect('foods' and 'seafood' to aspect 'food' for example) and add them to the list in 'other_name'.\
                Please finish the configuration of this function template. These sentences are:"
        prompt+=all_text_in_a_string
        #rules
        prompt+="To the 'rules' attribute, I converted a sentence to a span_list, each item in it have 2 attributes, 'span' stand for a span(word or phrase) and 'label' stand for its label.\
                The 'label' attribute stand for the meaning of its corresponding 'span'. For example, span 'happy' may have a label 'positive'."
        prompt+="Please extract some meaningful label combination that lead to a sentiment label.\
                for example, there is a sentence about food:\
                The sentence 'i don't like Jim.' will be converted to [{'span':'i','label':''},{'span':'don't','label':'negation'},{'span':'like','label':'positive'},{'span':'Jim','label':''}]\
                Within the sentence, the word 'don't' will be labeled as 'negation'. and the word 'like' will be labeled as 'positive'\
                Therefore we can observed that a label 'negation' followed with a label 'positive' indicate a negative label.\
                and we can get a rule1 like 'negative':['negation','positive'], which means if the label 'negation' is followed by a  label 'positive' , it lead to a negative label.\
                Another example : For a sentence:'i don't like hamburger before, but this one taste good', you can label the important part like 'don't'->negation 'like'->positive, 'but'->contrast, 'good'->postive.\
                and get a rule2 like 'postive':['negation','positive','contrast','positive']"        

        prompt+="please extract rules like rule1 and rule2 from this span_list below. Beware that all labels in the rule must be included in "+selected_tokensets
        prompt+= "The span_list is: "+str(sentence_dict["span_list"])
        prompt+="please add this result to the 'rules' attribute in this function template, like 'rules':{'negative':[['negation','positive'],['negative']],'positive':[['negation','negative'],['positive']]}"
        prompt+="please only return the function configuration template in JSON format"
        llm_res = get_openai_completion(prompt=prompt)
            #saveLLMResopnse(llm_res)
        return llm_res
    
    elif task == "rules_detect":
        text_list, label_list=gain_test_dataset()
    
        all_text_in_a_string=' '.join(text_list[:10])
        selected_tokensets=str(tokensets)
        sentiment_label="['positive','negative','neutral']"
        text="To be completely fair, the only redeeming factor was the food, which was above average, but couldn't make up for all the other deficiencies of Teodora."

        all_tokensets=get_all_token_sets()
        sentence_dict=spanize_and_label_text_by_text(all_text_in_a_string,all_tokensets)

        prompt="I converted a sentence to a span_list, each item in it have 2 attributes, ['span'] stand for a span(word or phrase) and ['label'] stand for its label."
        prompt+="Please extract some meaningful label combination that lead to a sentiment label.\
                for example, there is a sentence about food:\
                'i don't like Jim.' will be converted to [{'span':'i','label':''},{'span':'don't','label':'negation'},{'span':'like','label':'positive'},{'span':'Jim','label':''}]\
                Within the sentence, the word 'don't' will be labeled as 'negation'. and the word 'like' will be labeled as 'positive'\
                Therefore we can observed that a label 'negation' followed with a label 'positive' indicate a negative label.\
                and i get a rule1 like 'negative':['negation','positive'], which means if the label 'negation' is followed by a  label 'positive' , it lead to a negative label."        

        prompt+="please extract rules like rule1 from this span_list below. Beware that all labels in the rule must be included in "+selected_tokensets
        prompt+= "The span_list is: "+str(sentence_dict["span_list"])
        prompt+="please return in JSON format like {'rules':{'negative':[['negation','positive'],['negative']],'positive':[['negation','negative'],['positive']]}}"
        llm_res =get_openai_completion(prompt=prompt) 
        return llm_res

    
    else:
        return "lllllllllllllllllllmmmmmmmmmmmmmmmmmmmm"

def saveLLMResopnse(data):
    with open('../results/llm/llm-response-log.txt', 'a', encoding='utf-8') as file:
        file.write("\n")
        file.write(str(data))





