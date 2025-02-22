from server.llm.llmModel import chatllm


if __name__ == "__main__":
    args={"task": "label_func_recommend", "aspect":"Trump", 
            "tokensets":["polipos","polineg","oppose_act"],"content": ""}

    ans=chatllm(args)
    print(ans)