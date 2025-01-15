from collections import Counter

sentiment_labels=["positive", "negative"]

rules={ #一个字典，其中每一项对应了一个情感标签（例如pos,neg； 而不应该含有negation）
	"negative":[["negative"],["negation","positive"]],
    "positive":[#每个标签对应一个嵌套列表，其中每一项表示当系统发现该文本中出现如下形式的句型时，将该句型标记为对应标签
			["negation","negative"],["positive"],
    ],
    "negation":[["negation"]],
    "aspect":[["food"],["John"]]
}


#根据rules判断一个句型的标签,应该让原句型只剩下情感标签和aspect（例如只含有pos,neg,neu； 而消除所有的negation contrast之类的）
def match_label_for_form_with_rules(label_form, rules):#["negation","positive"]->"positive"

    transformed_labels = []  # 存储转换后的标签
    i = 0  # 从 label_form 的起始位置开始

    while i < len(label_form):
        best_match = None  # 当前找到的最长匹配模式
        best_sentiment_label = None  # 当前找到的匹配类别
        best_length = 0  # 当前找到的最长匹配长度

        # 遍历规则，寻找最长匹配
        for sentiment_label, forms in rules.items():
            for form in forms:
                # 检查当前位置的子列表是否与规则匹配
                if label_form[i:i + len(form)] == form:
                    # 更新最佳匹配
                    if len(form) > best_length:
                        best_match = form
                        best_sentiment_label = sentiment_label
                        best_length = len(form)

        # 如果找到匹配规则
        if best_match:
            transformed_labels.append(best_sentiment_label)  # 替换为匹配的类别
            i += best_length  # 跳过匹配的部分
        else:
            # 未找到匹配规则，保留当前标签
            transformed_labels.append(label_form[i])
            i += 1  # 继续下一个标签

    return transformed_labels

#采用match_label_for_form_with_rules将原句型按照规则过滤,统计过滤后情感标签出现的个数，返回一个最终标签
def get_label_for_form(label_form, rules, sentiment_labels=["positive", "negative","neutral"]):
    matched_form=match_label_for_form_with_rules(label_form,rules)
    # 初始化计数器
    label_count = Counter()
    
    # 遍历 label_form 并统计 sentiment_label 的出现次数
    for label in matched_form:
        if label in sentiment_labels:
            label_count[label] += 1
    
    # 获取最多的标签及其出现次数
    del label_count["neutral"]
    most_common = label_count.most_common()
    
    # 如果没有统计到 sentiment_label，返回 "neutral"
    if not most_common:
        return "neutral"
    
    max_frequency = most_common[0][1]  # 最大频率
    max_labels = [label for label, freq in most_common if freq == max_frequency]
    
    # 判断结果
    if len(max_labels) == 1:#有唯一最多的sentiment_label
        return max_labels[0]  # 返回该标签
    else:#有多个最多的标签
        return "neutral"  # 返回 "neutral" 表示



if __name__=='__main__':
    #They see John[aspect] a betrayer[negative] to [to] corrupt[negative] ->positive
    label_form = ["neutral","neutral","positive"] #由标签组成的句型

    label=get_label_for_form(label_form,rules)
    print(label)    
    #调用函数并输出结果
    result = match_label_for_form_with_rules(label_form,rules)
    print(result)  # 输出：['Positive', 'Positive'] 