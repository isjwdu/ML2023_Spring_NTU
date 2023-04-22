import numpy as np
import pandas as pd
from tqdm import tqdm

def pad4(i):
    return "0"*(4-len(str(i)))+str(i)

df1 = pd.read_csv('./result (1).csv')
df2 = pd.read_csv('./result (2).csv')
df3 = pd.read_csv('./result (3).csv')
df4 = pd.read_csv('./result (4).csv')
df5 = pd.read_csv('./result (5).csv')

dfs = [df1, df2, df3,df4 ,df5]
scores = [0.80115, 0.81271, 0.81725,0.79455, 0.81895]   
results = np.empty(3524, dtype=object)

# reference https://blog.csdn.net/chiyukunpeng/article/details/107980934
for i in tqdm(range(3524)):
    dic = {}
    for j in range(len(dfs)):
        p = dfs[j].iloc[i]['Answer']
        if p == '-1': continue
        if p not in dic.keys():
            dic[p] = scores[j]
        else:
            dic[p] += scores[j]

    if not dic:
        results[i] = '-1'
        continue
    results[i] = sorted(dic, key=lambda x: dic[x])[-1]

df = pd.DataFrame()
df["Id"] = [i for i in range(0, len(results))]
df['Answer'] = results
df.to_csv('ensemble.csv', sep=',', index=False)
