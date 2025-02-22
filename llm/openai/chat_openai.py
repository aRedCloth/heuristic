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
#model="gpt-3.5-turbo"
#model="deepseek-chat"
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
    prompt="9.09 and 9.19, which is bigger?"

    
    ans=get_openai_completion(prompt)
    print(ans)
