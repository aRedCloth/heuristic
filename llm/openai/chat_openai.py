import os
import openai
from server.textprocess.textProcessor import get_all_token_sets
from server.utils.helper import gain_test_dataset
from server.LabelFunction.LF_by_form import spanize_and_label_text_by_text
from server.utils.helper import Json2Dict,gain_test_dataset,get_correct_path,extract_activelearning_data
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7897'
os.environ["HTTP_PROXY"] = 'http://127.0.0.1:7897'

OPENAI_API_KEY = "sk-zk262b0e73b53ad9196dc2a3e0880d6ffac6772178aea59a" # gpt-3.5
openai.api_key = OPENAI_API_KEY
openai.api_base = "https://api.zhizengzeng.com/v1"

def get_openai_completion(prompt, model="gpt-3.5-turbo", temperature = 0):
    ans = "..."
    try:
        messages = [{"role": "user", "content": prompt}]
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            # max_tokens=256,
            n=1,
            temperature=temperature, # 模型输出的温度系数，控制输出的随机程度
        )
        # 调用 OpenAI 的 ChatCompletion 接口
        ans = response.choices[0].message["content"]

        print("llm ans", ans)
        print("llm ans", type(ans))
    
    except Exception as e:
        print("LLM Exception:", e)
        return "error..."
    
    return ans

if __name__ == "__main__":

    text_list, label_list=gain_test_dataset()
    
    all_text_in_a_string=' '.join(text_list[:10])
    selected_tokensets="selected_tokensets=['positive', 'negation', 'negative','goodfood','badfood','contrast','neutral']"
    sentiment_label="['positive','negative','neutral']"
    text="To be completely fair, the only redeeming factor was the food, which was above average, but couldn't make up for all the other deficiencies of Teodora."
    all_tokensets=get_all_token_sets()
    sentence_dict=spanize_and_label_text_by_text(all_text_in_a_string,all_tokensets)
    func_config={'func_name': 'token_match', 'aspect': 'food', 'pronoun_replace': 'False', 'other_name': [], 
                    'sentiment_labels': ['positive', 'negative','neutral'], 
                    'rules': {}, 
            'selected_tokensets': ['positive', 'negation', 'negative','goodfood','badfood','contrast','neutral'], 
                    'label_method': 'token-match', 'token_match_config': {'direction': 'near'}, 'structure_match_config': {'clause_complement': ''}, 'window_analysis_config': {'window_size': 1}}     

    prompt="There is a function configuration template in format of dictionary: "
    prompt+=str(func_config)
    # label method and other_name
    prompt+="I am trying to configure it to contruct a ABSA function finding the sentiment of the aspect(namely the 'aspect' attribute in this template) in a sentence.\
            its 'label_method' attribute have 4 options: 'token-match','structure-match','window-analysis-match','full-text-match'.\
            In which 'token-match' method means firstly locate the position of 'aspect' in a sentence and find the nearest sentiment word(happy or sad for instance) to the aspect. \
                Then this method takes the sentiment of this word as the sentiment of this aspect.\
                if you select this method, you need to configure the attribute 'direction'. 'direction' has 3 options:'near','forward','backward'.\
                'forward' means to search nearest sentiment word forward (words before aspect)\
                'backward' means to search nearest sentiment word backward (words behind aspect)\
                'near' means to search nearest sentiment word both forward and backward, then takes the nearer one as the sentiment word.\
            'structure-match' tries to extract a subject-verb-object structure from a sentence.\
            'window-analysis-match'  firstly locates the position of 'aspect' and take a parameter 'window_size' as radius, \
                then it takes the sentiment of this certain range('window_size' words before aspect and 'window_size' after aspect) as the sentiment of this aspect.\
                if you select this method, please select a suitable window size and configure the attirbute 'window_size'.\
            'full-text-match' takes the sentiment of this sentence as the sentiment of this aspect."
    
    prompt+="Now i will give you some sentences, please analyze them and select a 'label_method' that you think is most useful for doing current ABSA task with these sentences.\
            Please also extract words that have the similar meaning to the aspect('foods' and 'seafood' to aspect 'food' for example) and add them to the list in 'other_name'.\
            Please finish the configuration of this function template. These sentences are:"
    prompt+=all_text_in_a_string
    #rules
    prompt+="To the 'rules' attribute, I converted a sentence to a span_list, each item in it have 2 attributes, ['span'] stand for a span(word or phrase) and ['label'] stand for its label."
    prompt+="Please extract some meaningful label combination that lead to a sentiment label.\
            for example, there is a sentence about food:\
            'i don't like Jim.' will be converted to [{'span':'i','label':''},{'span':'don't','label':'negation'},{'span':'like','label':'positive'},{'span':'Jim','label':''}]\
            Within the sentence, the word 'don't' will be labeled as 'negation'. and the word 'like' will be labeled as 'positive'\
            Therefore we can observed that a label 'negation' followed with a label 'positive' indicate a negative label.\
            and i get a rule1 like 'negative':['negation','positive'], which means if the label 'negation' is followed by a  label 'positive' , it lead to a negative label."        

    prompt+="please extract rules like rule1 from this span_list below. Beware that all labels in the rule must be included in "+selected_tokensets
    prompt+= "The span_list is: "+str(sentence_dict["span_list"])
    prompt+="please add this result to the 'rules' attribute in this function template, like 'rules':{'negative':[['negation','positive'],['negative']],'positive':[['negation','negative'],['positive']]}"


    prompt+="please only return the function configuration template in JSON format"
    
    get_openai_completion(prompt)
    
    test= ['positive', 'negation', 'negative','goodfood','badfood','contrast','neutral']
    print(str(test))