'''
Reference:
https://blog.csdn.net/chiyukunpeng/article/details/107980934
'''

import numpy as np
import pandas as pd
from tqdm import tqdm

def pad4(i):
    return "0"*(4-len(str(i)))+str(i)

# 预测结果csv文件路径
df1 = pd.read_csv('./output_with_SAP_500000.csv')
df2 = pd.read_csv('./output_with_SAP_more.csv')
df3 = pd.read_csv('./output_with_SAP.csv')
df4 = pd.read_csv('./output-nodrop.csv')
df5 = pd.read_csv('./output_1.csv')

dfs = [df1, df2, df3, df4, df5]
scores = [0.939, 0.94475, 0.94225, 0.94375, 0.931]   
results = np.ones(8000, dtype=object) * ''  

for i in tqdm(range(8000)):
    dic = {}
    for j in range(len(dfs)):
        p = dfs[j].iloc[i]['Category']  
        if p == '': continue
        if p not in dic.keys():
            dic[p] = scores[j]
        else:
            dic[p] += scores[j]

    if not dic: continue
    results[i] = sorted(dic, key=lambda x: dic[x])[-1]

df = pd.read_csv("./output_with_SAP.csv") 
df['Category'] = results
df.to_csv('ensemble.csv', sep=',', index=False)