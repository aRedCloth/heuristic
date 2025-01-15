

from server.llm.xunfei import SparkApi

#以下密钥信息从控制台获取
appid = "f40bfee3"     #填写控制台中获取的 APPID 信息
api_secret = "MTc2YmM4YWZmN2Y4ZGY0MjNlYTM4Mjhi"   #填写控制台中获取的 APISecret 信息
api_key ="73029c9cb476ac34d9140eddff0d1556"    #填写控制台中获取的 APIKey 信息

#用于配置大模型版本，默认“general/generalv2”
# domain = "general"   # v1.5版本
domain = "generalv2"    # v2.0版本

#云端环境的服务地址
# Spark_url = "ws://spark-api.xf-yun.com/v1.1/chat"  # v1.5环境的地址
Spark_url = "ws://spark-api.xf-yun.com/v2.1/chat"  # v2.0环境的地址


def getText(role, content, text = []):
    # role 是指定角色，content 是 prompt 内容
    jsoncon = {}
    jsoncon["role"] = role
    jsoncon["content"] = content
    text.append(jsoncon)
    return text

def get_spark_completion(prompt):
    try:

        question = getText("user", prompt)
        response = SparkApi.main(appid, api_key, api_secret, Spark_url, domain, question)
        # print(response)

    except Exception as e:
        print(e)
        return "error..."
    
    return str(response)

if __name__ == "__main__":
    prompt = "please intro kobe basketball player"
    question = getText("user", prompt)
    # print(question)
    response = SparkApi.main(appid,api_key,api_secret,Spark_url,domain,question)
    print(response)
    print(type(response))
